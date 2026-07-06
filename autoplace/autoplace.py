from kipy import KiCad
from kipy.geometry import Vector2, Angle
from kipy.board_types import (
    BoardLayer,
    to_concrete_board_shape,
)
from dataclasses import dataclass, field
import numpy as np
import torch
import wx


@dataclass
class Pin:
    local_pos: np.ndarray
    net: str


@dataclass
class Component:
    id: str
    pos: np.ndarray
    theta: float = 0.0
    half_size: np.ndarray = field(default_factory=lambda: np.array([2.5, 2.5]))
    anchor_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pins: list = field(default_factory=list)
    fixed: bool = False
    flipped: bool = False  # on B.Cu


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])  # KiCad's Y-down rotation


def pin_world(comp, pin):
    return comp.pos + rot(comp.theta) @ pin.local_pos


def rebuild_nets(components):
    nets = {}
    for cid, comp in components.items():
        for pi, pin in enumerate(comp.pins):
            nets.setdefault(pin.net, []).append((cid, pi))
    return nets


def hpwl(components):
    total = 0.0
    for members in rebuild_nets(components).values():
        if len(members) < 2:
            continue
        pts = np.array(
            [pin_world(components[c], components[c].pins[p]) for c, p in members]
        )
        total += np.ptp(pts[:, 0]) + np.ptp(pts[:, 1])
    return total


def n2m(v):
    NM = 1e6  # nanometres per millimetre
    return np.array([v.x, v.y]) / NM


def rect(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])


def get_edge_cuts(board):
    box = None
    for s in board.get_shapes():
        if s.layer == BoardLayer.BL_Edge_Cuts:
            b = to_concrete_board_shape(s).bounding_box()
            box = (
                b if box is None else (box.merge(b) or box)
            )  # merge mutates, returns None
    if box is None:
        return None
    lo = n2m(box.pos)
    return rect(lo, lo + n2m(box.size))


def get_components(board):
    def calc_geometry(board, fp):
        CRTYD = (BoardLayer.BL_F_CrtYd, BoardLayer.BL_B_CrtYd)
        box = None
        for s in fp.definition.shapes:
            if s.layer in CRTYD:
                b = to_concrete_board_shape(s).bounding_box()
                box = b if box is None else (box.merge(b) or box)
        if box is None:
            # raise ValueError()
            box = board.get_item_bounding_box(fp)
        if box is None:
            raise ValueError()
        center = n2m(box.pos) + n2m(box.size) / 2  # world center of the body
        hw = n2m(box.size) / 2
        c, s = (
            abs(np.cos(fp.orientation.to_radians())),
            abs(np.sin(fp.orientation.to_radians())),
        )
        half = (
            np.array([hw[1], hw[0]]) if s > c else hw
        )  # world AABB -> local (90/270 swap)
        return center, half

    components = {}
    for fp in board.get_footprints():
        ref = fp.reference_field.text.value
        anchor = n2m(fp.position)
        theta = fp.orientation.to_radians()
        Rinv = rot(-theta)
        center, half = calc_geometry(board, fp)
        pins = [
            Pin(
                local_pos=Rinv @ (n2m(p.position) - center), net=p.net.name
            )  # pins rel. to CENTER
            for p in fp.definition.pads
        ]
        components[ref] = Component(
            id=ref,
            pos=center,
            theta=theta,
            half_size=half,
            pins=pins,
            fixed=fp.locked,
            flipped=(fp.layer == BoardLayer.BL_B_Cu),
            anchor_offset=Rinv
            @ (anchor - center),  # center->anchor vector, neutral frame
        )
    return components


class Cancel(Exception):
    pass


def place(
    components,
    edges,
    n_iters=1000,
    lr_pos=None,
    lr_theta=0.05,
    lr_side=0.05,
    gamma=1.0,
    M=64,
    lam_max=50.0,
    verbose=True,
    on_step=None,
):
    """Optimize positions + rotations + sides in place. Returns (components, history)."""
    ids = list(components)
    hs = torch.tensor(
        np.array([components[c].half_size for c in ids]), dtype=torch.float32
    )
    pos = torch.tensor(
        np.array([components[c].pos for c in ids]),
        dtype=torch.float32,
        requires_grad=True,
    )
    theta = torch.tensor(
        [components[c].theta for c in ids], dtype=torch.float32, requires_grad=True
    )
    fixed = torch.tensor([components[c].fixed for c in ids])
    fixed_pos = pos.detach().clone()
    fixed_theta = theta.detach().clone()

    f = torch.tensor([float(components[c].flipped) for c in ids])
    z = (2 * f - 1) * 1.5  # sigmoid(+-1.5) ~ 0.18/0.82: committed but persuadable
    z[fixed] *= 6.0  # locked parts: effectively hard 0/1
    z.requires_grad_(True)
    z0 = z.detach().clone()

    lo = torch.tensor(edges[0], dtype=torch.float32)
    hi = torch.tensor(edges[2], dtype=torch.float32)
    if lr_pos is None:
        lr_pos = 0.005 * float((hi - lo).max())  # step ~0.5% of board per iter
    part_area = float((4 * hs[:, 0] * hs[:, 1]).sum())
    tol = 0.02 * part_area  # tolerate 2% overlapped area

    net_pins = []
    for members in rebuild_nets(components).values():
        if len(members) < 2:
            continue
        net_pins.append(
            [
                (
                    ids.index(cid),
                    torch.tensor(
                        components[cid].pins[pi].local_pos, dtype=torch.float32
                    ),
                )
                for cid, pi in members
            ]
        )

    def pin_pos(i, off, mx):
        c, s = torch.cos(theta[i]), torch.sin(theta[i])
        ox = off[0] * mx[i]  # mirror in the local frame, then rotate
        return pos[i] + torch.stack(
            [c * ox + s * off[1], -s * ox + c * off[1]]
        )  # [[c,s],[-s,c]]

    def hs_eff():
        c, s = torch.cos(theta).abs(), torch.sin(theta).abs()
        return torch.stack(
            [c * hs[:, 0] + s * hs[:, 1], c * hs[:, 1] + s * hs[:, 0]], dim=1
        )

    def wa_wirelength(mx):
        total = 0.0
        for members in net_pins:
            pts = torch.stack([pin_pos(i, off, mx) for i, off in members])
            for d in range(2):
                x = pts[:, d]
                total = (
                    total
                    + (x * torch.softmax(x / gamma, 0)).sum()
                    - (x * torch.softmax(-x / gamma, 0)).sum()
                )
        return total

    bx = torch.linspace(lo[0], hi[0], M + 1)
    by = torch.linspace(lo[1], hi[1], M + 1)
    bin_area = (bx[1] - bx[0]) * (by[1] - by[0])

    def density_map(side):
        h = hs_eff()
        ox = (
            torch.minimum(pos[:, 0:1] + h[:, 0:1], bx[None, 1:])
            - torch.maximum(pos[:, 0:1] - h[:, 0:1], bx[None, :-1])
        ).clamp(min=0)
        oy = (
            torch.minimum(pos[:, 1:2] + h[:, 1:2], by[None, 1:])
            - torch.maximum(pos[:, 1:2] - h[:, 1:2], by[None, :-1])
        ).clamp(min=0)
        return torch.einsum("sn,ni,nj->sij", side, ox, oy) / bin_area  # (2, M, M)

    def electro_energy(rho):
        q = (rho - 1.0).clamp(min=0)  # only over-capacity density is charge
        q = q - q.mean(dim=(-2, -1), keepdim=True)  # neutralize per side
        fx = torch.fft.fftfreq(M) * 2 * np.pi
        k2 = fx[:, None] ** 2 + fx[None, :] ** 2
        k2[0, 0] = 1.0
        phi = torch.fft.ifft2(torch.fft.fft2(q) / k2).real
        return 0.5 * (q * phi).sum()

    opt = torch.optim.Adam(
        [
            {"params": [pos], "lr": lr_pos},
            {"params": [theta], "lr": lr_theta},
            {"params": [z], "lr": lr_side},
        ]
    )
    lam, snap_w, bin_w = 1e-3, 0.0, 0.0
    history = {"wl": [], "den": [], "overflow": [], "lam": []}

    for it in range(n_iters):
        opt.zero_grad()
        s = torch.sigmoid(z)  # occupancy of B.Cu
        m = s + f * (1 - 2 * s)  # prob. of sitting opposite the imported side
        mx = 1 - 2 * m  # +1 as-imported ... -1 mirrored
        side = torch.stack([1 - s, s])  # (2, N) front/back weights
        wl = wa_wirelength(mx)
        den = lam * electro_energy(density_map(side))
        snap = snap_w * torch.sin(2 * theta).pow(2).sum()
        binar = bin_w * (s * (1 - s)).sum()  # anneal s to 0/1, like the theta snap
        (wl + den + snap + binar).backward()
        opt.step()
        with torch.no_grad():
            pos.data[fixed] = fixed_pos[fixed]
            theta.data[fixed] = fixed_theta[fixed]
            z.data[fixed] = z0[fixed]
            h = hs_eff()
            pos.data = torch.max(torch.min(pos, hi - h), lo + h)
            s_ = torch.sigmoid(z)
            overflow = (
                (density_map(torch.stack([1 - s_, s_])) - 1.0).clamp(min=0).sum()
                * bin_area
            ).item()
            amb = int(((s_ > 0.1) & (s_ < 0.9)).sum())  # parts still undecided
        if overflow > tol:
            lam = min(lam * 1.03, lam_max)
        elif overflow < 0.5 * tol:
            lam *= 0.97
        if it > 0.4 * n_iters:
            snap_w = min(snap_w + 0.02, 5.0)
            bin_w = min(bin_w + 0.02, 5.0)
        history["wl"].append(wl.item())
        history["den"].append(den.item())
        history["overflow"].append(overflow)
        history["lam"].append(lam)
        if on_step is not None and on_step(it, n_iters, history) is False:
            raise Cancel
        if verbose and it % 100 == 0:
            print(
                f"it {it:4d}  WL {wl.item():9.2f}  lam*density {den.item():8.2f}  "
                f"lam {lam:.4f}  overflow {overflow:7.2f}  undecided {amb}"
            )

    with torch.no_grad():  # legalize rotation + side
        theta.data = torch.round(theta / (np.pi / 2)) * (np.pi / 2)
        theta.data[fixed] = fixed_theta[fixed]
        back = torch.sigmoid(z) > 0.5
        back[fixed] = f[fixed] > 0.5
        h = hs_eff()
        pos.data = torch.max(torch.min(pos, hi - h), lo + h)

    p, t = pos.detach().numpy(), theta.detach().numpy()
    MIR = np.diag([-1.0, 1.0])
    for i, cid in enumerate(ids):
        comp = components[cid]
        comp.pos = p[i].astype(float)
        comp.theta = float(t[i])
        if bool(back[i]) != comp.flipped:  # bake the mirror into the local frame
            for pin in comp.pins:
                pin.local_pos = MIR @ pin.local_pos
            comp.anchor_offset = MIR @ comp.anchor_offset
            comp.flipped = bool(back[i])
    return components, history


def legalize(components, edges, iters=300):
    ids = list(components)
    lo, hi = np.asarray(edges[0], float), np.asarray(edges[2], float)

    def eff_half(c):  # rotated AABB half-extent, matches the placer's hs_eff
        ct, st = abs(np.cos(c.theta)), abs(np.sin(c.theta))
        return np.array(
            [
                ct * c.half_size[0] + st * c.half_size[1],
                ct * c.half_size[1] + st * c.half_size[0],
            ]
        )

    for _ in range(iters):
        moved = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = components[ids[i]], components[ids[j]]
                if a.flipped != b.flipped:
                    continue
                d = b.pos - a.pos
                pen = (eff_half(a) + eff_half(b)) - np.abs(d)  # per-axis penetration
                if np.all(pen > 0):  # boxes intersect
                    ax = int(np.argmin(pen))  # least-penetration axis
                    s = np.sign(d[ax]) or 1.0
                    push = (pen[ax] / 2 + 1e-3) * s
                    if not a.fixed:
                        a.pos[ax] -= push
                    if not b.fixed:
                        b.pos[ax] += push
                    moved = True
        for cid in ids:  # keep inside board
            c = components[cid]
            if not c.fixed:
                c.pos = np.clip(c.pos, lo + eff_half(c), hi - eff_half(c))
        if not moved:
            break
    return components


def frame_angle(A, B):
    """Rotation phi minimizing sum |A_i - rot(phi) @ B_i| over matched points (2D Kabsch)."""
    A, B = np.asarray(A, float).reshape(-1, 2), np.asarray(B, float).reshape(-1, 2)
    if len(A) == 0:
        return 0.0
    D = float((A * B).sum())
    C = float((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum())
    return float(np.arctan2(C, D)) if (C or D) else 0.0


def write_back(kicad, board, components, message="autoplace"):
    # The IPC API cannot flip by writing fp.layer: FOOTPRINT::Deserialize calls
    # SetLayer (a relabel), never Flip. So side changes go through the real flip
    # action on a selection; then re-read the truly mirrored geometry and solve
    # for the pose that lands it where the model wants it.
    fps = {fp.reference_field.text.value: fp for fp in board.get_footprints()}
    to_flip = [
        ref
        for ref, comp in components.items()
        if ref in fps
        and not comp.fixed
        and comp.flipped != (fps[ref].layer == BoardLayer.BL_B_Cu)
    ]
    if to_flip:
        board.clear_selection()
        board.add_to_selection([fps[r] for r in to_flip])
        status = kicad.run_action("pcbnew.InteractiveEdit.flip").status
        board.clear_selection()
        fresh = get_components(board)
        for ref in to_flip:
            comp, fc = components[ref], fresh[ref]
            if fc.flipped != comp.flipped:
                raise RuntimeError(f"flip didn't take on {ref} (status {status})")
            phi = frame_angle(
                [p.local_pos for p in comp.pins], [p.local_pos for p in fc.pins]
            )  # rotation aligning KiCad's post-flip frame to the model's baked frame
            comp.theta += phi  # theta absorbs the frame difference...
            comp.pins = fc.pins  # ...so we can adopt KiCad's own frame verbatim
            comp.anchor_offset = fc.anchor_offset

    commit = board.begin_commit()
    try:
        changed = []
        for fp in board.get_footprints():
            comp = components.get(fp.reference_field.text.value)
            if comp is None or comp.fixed:  # skip locked / unmatched
                continue
            # comp.pos is the body CENTER; KiCad pivots on the anchor, so convert back
            anchor = comp.pos + rot(comp.theta) @ comp.anchor_offset
            fp.position = Vector2.from_xy_mm(float(anchor[0]), float(anchor[1]))
            fp.orientation = Angle.from_degrees(float(np.degrees(comp.theta)))
            changed.append(fp)
        board.update_items(changed)
        board.push_commit(commit, message)
        return changed
    except Exception:
        board.drop_commit(commit)  # roll back, don't leave a commit open
        raise


if __name__ == "__main__":
    ITERS = 1000
    try:
        kicad = KiCad()
        print(f"Connected to KiCad {kicad.get_version()}")
    except BaseException as e:
        print(f"Not connected to KiCad: {e}")
        exit(1)

    board = kicad.get_board()
    footprints = board.get_footprints()

    components, edges = get_components(board), get_edge_cuts(board)
    print(f"{len(components)} components, initial HPWL = {hpwl(components):.1f}")

    app = wx.App(False)
    dlg = wx.ProgressDialog(
        "autoplace",
        "starting…",
        maximum=ITERS,
        style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME | wx.PD_CAN_ABORT,
    )

    def on_step(it, n, hist):
        cont, _ = dlg.Update(
            it + 1,
            f"{it + 1}/{n}   WL {hist['wl'][-1]:.0f}   overflow {hist['overflow'][-1]:.1f}",
        )
        return cont  # Cancel button -> False

    try:
        place(components, edges, n_iters=ITERS, on_step=on_step)
        legalize(components, edges)
        write_back(kicad, board, components)
    except Cancel:
        pass  # don't writeback

    dlg.Destroy()
