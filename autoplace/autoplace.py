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
    gamma=1.0,
    M=64,
    lam_max=50.0,
    verbose=True,
    on_step=None,
):
    """Optimize positions + rotations in place. Returns (components, history)."""
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

    def pin_pos(i, off):
        c, s = torch.cos(theta[i]), torch.sin(theta[i])
        return pos[i] + torch.stack(
            [c * off[0] + s * off[1], -s * off[0] + c * off[1]]
        )  # [[c,s],[-s,c]]

    def hs_eff():
        c, s = torch.cos(theta).abs(), torch.sin(theta).abs()
        return torch.stack(
            [c * hs[:, 0] + s * hs[:, 1], c * hs[:, 1] + s * hs[:, 0]], dim=1
        )

    def wa_wirelength():
        total = 0.0
        for members in net_pins:
            pts = torch.stack([pin_pos(i, off) for i, off in members])
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

    def density_map():
        h = hs_eff()
        ox = (
            torch.minimum(pos[:, 0:1] + h[:, 0:1], bx[None, 1:])
            - torch.maximum(pos[:, 0:1] - h[:, 0:1], bx[None, :-1])
        ).clamp(min=0)
        oy = (
            torch.minimum(pos[:, 1:2] + h[:, 1:2], by[None, 1:])
            - torch.maximum(pos[:, 1:2] - h[:, 1:2], by[None, :-1])
        ).clamp(min=0)
        return torch.einsum("ni,nj->ij", ox, oy) / bin_area

    def electro_energy(rho):
        q = (rho - 1.0).clamp(min=0)  # only over-capacity density is charge
        q = q - q.mean()  # neutralize for the periodic solve
        fx = torch.fft.fftfreq(M) * 2 * np.pi
        k2 = fx[:, None] ** 2 + fx[None, :] ** 2
        k2[0, 0] = 1.0
        phi = torch.fft.ifft2(torch.fft.fft2(q) / k2).real
        return 0.5 * (q * phi).sum()

    opt = torch.optim.Adam(
        [{"params": [pos], "lr": lr_pos}, {"params": [theta], "lr": lr_theta}]
    )
    lam, snap_w = 1e-3, 0.0
    history = {"wl": [], "den": [], "overflow": [], "lam": []}

    for it in range(n_iters):
        opt.zero_grad()
        wl = wa_wirelength()
        den = lam * electro_energy(density_map())
        snap = snap_w * torch.sin(2 * theta).pow(2).sum()
        (wl + den + snap).backward()
        opt.step()
        with torch.no_grad():
            pos.data[fixed] = fixed_pos[fixed]
            theta.data[fixed] = fixed_theta[fixed]
            h = hs_eff()
            pos.data = torch.max(torch.min(pos, hi - h), lo + h)
            overflow = ((density_map() - 1.0).clamp(min=0).sum() * bin_area).item()
        if overflow > tol:
            lam = min(lam * 1.03, lam_max)
        elif overflow < 0.5 * tol:
            lam *= 0.97
        if it > 0.4 * n_iters:
            snap_w = min(snap_w + 0.02, 5.0)
        history["wl"].append(wl.item())
        history["den"].append(den.item())
        history["overflow"].append(overflow)
        history["lam"].append(lam)
        if on_step is not None and on_step(it, n_iters, history) is False:
            raise Cancel
        if verbose and it % 100 == 0:
            print(
                f"it {it:4d}  WL {wl.item():9.2f}  lam*density {den.item():8.2f}  "
                f"lam {lam:.4f}  overflow {overflow:7.2f}"
            )

    with torch.no_grad():  # legalize rotation
        theta.data = torch.round(theta / (np.pi / 2)) * (np.pi / 2)
        theta.data[fixed] = fixed_theta[fixed]
        h = hs_eff()
        pos.data = torch.max(torch.min(pos, hi - h), lo + h)

    p, t = pos.detach().numpy(), theta.detach().numpy()
    for i, cid in enumerate(ids):
        components[cid].pos = p[i].astype(float)
        components[cid].theta = float(t[i])
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


def write_back(board, components, message="autoplace"):
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
        write_back(board, components)
    except Cancel:
        pass  # don't writeback

    dlg.Destroy()
