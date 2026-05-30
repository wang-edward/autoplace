"""Force-directed PCB autoplacer.

Overdamped descent on per-net springs (applied at pins, so torque rotates parts)
plus AABB repulsion, with periodic Hungarian-based pin-swap optimization.
Constraints: per-component allowed_rect or allowed_polygon, plus a discrete
set of allowed orientations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches

QUAD = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]


# ---------- Data ----------


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
    pins: list = field(default_factory=list)
    fixed: bool = False
    allowed_rect: tuple | None = None
    allowed_polygon: np.ndarray | None = None
    allowed_orientations: list | None = None


@dataclass
class SwapGroup:
    component_id: str
    pin_indices: list


# ---------- Geometry ----------


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def pin_world(comp, pin):
    return comp.pos + rot(comp.theta) @ pin.local_pos


def hs_rot(c):
    """Half-size after a 90-deg rotation snap; axes swap on odd quadrants."""
    return c.half_size[::-1] if round(c.theta / (np.pi / 2)) % 2 else c.half_size


def point_in_polygon(p, poly):
    inside, n, j = False, len(poly), len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > p[1]) != (yj > p[1])) and (
            p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def closest_on_polygon(p, poly):
    best, best_d = poly[0], np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = max(0.0, min(1.0, float((p - a) @ ab) / (float(ab @ ab) + 1e-12)))
        q = a + t * ab
        d = float((p - q) @ (p - q))
        if d < best_d:
            best_d, best = d, q
    return best


# ---------- Forces ----------


def attractive(components, nets):
    F_lin = {cid: np.zeros(2) for cid in components}
    F_tor = {cid: 0.0 for cid in components}
    for members in nets.values():
        if len(members) < 2:
            continue
        positions = [
            pin_world(components[cid], components[cid].pins[pi]) for cid, pi in members
        ]
        center = np.mean(positions, axis=0)
        for (cid, _), pos in zip(members, positions):
            f = center - pos
            F_lin[cid] += f
            r = pos - components[cid].pos
            F_tor[cid] += r[0] * f[1] - r[1] * f[0]
    return F_lin, F_tor


def repulsive(components, k=30.0, slop=0.4):
    F = {cid: np.zeros(2) for cid in components}
    ids = list(components.keys())
    for i in range(len(ids)):
        a = components[ids[i]]
        ah = hs_rot(a)
        for j in range(i + 1, len(ids)):
            b = components[ids[j]]
            bh = hs_rot(b)
            dx, dy = b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]
            ox = ah[0] + bh[0] + slop - abs(dx)
            oy = ah[1] + bh[1] + slop - abs(dy)
            if ox <= 0 or oy <= 0:
                continue
            if ox < oy:
                push = np.array([np.sign(dx or 1) * ox * k, 0.0])
            else:
                push = np.array([0.0, np.sign(dy or 1) * oy * k])
            F[ids[i]] -= push
            F[ids[j]] += push
    return F


# ---------- Constraints ----------


def project_position(comp):
    if comp.allowed_rect is not None:
        x0, y0, x1, y1 = comp.allowed_rect
        comp.pos[0] = np.clip(comp.pos[0], x0, x1)
        comp.pos[1] = np.clip(comp.pos[1], y0, y1)
        return
    poly = comp.allowed_polygon
    if poly is None:
        return
    if not point_in_polygon(comp.pos, poly):
        comp.pos = closest_on_polygon(comp.pos, poly)
    # Push body inside: any of the 4 AABB corners outside the polygon
    # gets projected back; max-magnitude per-axis push is applied to the center.
    for _ in range(6):
        hs = hs_rot(comp)
        corners = comp.pos + np.array(
            [[-hs[0], -hs[1]], [hs[0], -hs[1]], [hs[0], hs[1]], [-hs[0], hs[1]]]
        )
        push = np.zeros(2)
        moved = False
        for corner in corners:
            if point_in_polygon(corner, poly):
                continue
            d = closest_on_polygon(corner, poly) - corner
            if abs(d[0]) > abs(push[0]):
                push[0] = d[0]
            if abs(d[1]) > abs(push[1]):
                push[1] = d[1]
            moved = True
        if not moved:
            return
        comp.pos = comp.pos + push


def snap_orientation(comp, strength):
    """strength in [0, 1]; 1.0 is a hard snap."""
    if comp.allowed_orientations is None or strength <= 0:
        return
    deltas = [
        (a - comp.theta + np.pi) % (2 * np.pi) - np.pi
        for a in comp.allowed_orientations
    ]
    comp.theta += strength * min(deltas, key=abs)


# ---------- Pin swap (Hungarian) ----------


def rebuild_nets(components):
    nets = {}
    for cid, comp in components.items():
        for pi, pin in enumerate(comp.pins):
            nets.setdefault(pin.net, []).append((cid, pi))
    return nets


def optimize_swap_group(components, nets, group):
    comp = components[group.component_id]
    pins = [comp.pins[i] for i in group.pin_indices]
    nets_g = [p.net for p in pins]
    pos_g = [pin_world(comp, p) for p in pins]

    # Net centroid excluding the pins in this swap group (so it doesn't pull itself)
    centroids = []
    for net in nets_g:
        others = [
            pin_world(components[cid], components[cid].pins[pi])
            for cid, pi in nets[net]
            if not (cid == comp.id and pi in group.pin_indices)
        ]
        centroids.append(np.mean(others, axis=0) if others else comp.pos)

    n = len(pins)
    cost = np.array(
        [
            [float(np.sum((pos_g[i] - centroids[j]) ** 2)) for j in range(n)]
            for i in range(n)
        ]
    )
    _, col = linear_sum_assignment(cost)
    for i, j in enumerate(col):
        comp.pins[group.pin_indices[i]].net = nets_g[j]


# ---------- Main loop ----------


def run(components, swap_groups, n_iters=800, dt0=0.05, swap_every=25, snap_phase=0.4):
    dt = dt0
    for it in range(n_iters):
        nets = rebuild_nets(components)
        Fa, Tor = attractive(components, nets)
        Fr = repulsive(components)
        snap_str = max(0.0, (it / n_iters - snap_phase) / (1 - snap_phase)) ** 2

        for cid, comp in components.items():
            if comp.fixed:
                continue
            comp.pos = comp.pos + dt * (Fa[cid] + Fr[cid])
            # Single allowed orientation -> freeze rotation
            if (
                comp.allowed_orientations is not None
                and len(comp.allowed_orientations) == 1
            ):
                comp.theta = comp.allowed_orientations[0]
            else:
                comp.theta += dt * 0.15 * Tor[cid]
                snap_orientation(comp, snap_str * 0.5)
            project_position(comp)

        if it > 0 and it % swap_every == 0:
            for g in swap_groups:
                optimize_swap_group(components, nets, g)
        dt *= 0.996

    # Final hard snap + one more swap pass
    for comp in components.values():
        if not comp.fixed:
            snap_orientation(comp, 1.0)
            project_position(comp)
    nets = rebuild_nets(components)
    for g in swap_groups:
        optimize_swap_group(components, nets, g)


# ---------- Viz ----------


def draw(components, board, ax, title):
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.add_patch(
        patches.Polygon(
            board, facecolor="#f6f6e8", edgecolor="#444", linewidth=1.6, zorder=0
        )
    )
    for members in rebuild_nets(components).values():
        if len(members) < 2:
            continue
        pts = [
            pin_world(components[cid], components[cid].pins[pi]) for cid, pi in members
        ]
        # Skip dense power nets for visual clarity
        net = components[members[0][0]].pins[members[0][1]].net
        if net in ("GND", "VCC"):
            continue
        ctr = np.mean(pts, axis=0)
        for p in pts:
            ax.plot(
                [p[0], ctr[0]],
                [p[1], ctr[1]],
                "-",
                color="#6b6",
                alpha=0.5,
                linewidth=0.8,
            )
    for cid, c in components.items():
        box = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * c.half_size
        corners = (rot(c.theta) @ box.T).T + c.pos
        color = "#fcc" if cid.startswith("J_") else "#cce"
        ax.add_patch(
            patches.Polygon(
                corners, facecolor=color, edgecolor="#225", linewidth=1.0, zorder=2
            )
        )
        ax.text(
            c.pos[0], c.pos[1], cid, ha="center", va="center", fontsize=6.5, zorder=3
        )
        for p in c.pins:
            wp = pin_world(c, p)
            ax.plot(wp[0], wp[1], "o", color="#225", markersize=1.8, zorder=3)


# ---------- KiCad PCB import ----------
#
# KiCad uses S-expressions and a Y-down coordinate frame. We negate Y on read
# so the internal frame is Y-up (matches matplotlib and the rest of the code).
# Footprint bbox is estimated from pad extents + a small margin -- not perfect
# (real bbox is in the F.CrtYd courtyard layer) but adequate for placement.
# Edge.Cuts arcs are not handled; gr_line and gr_rect only.


def parse_sexp(text):
    tokens, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            tokens.append(text[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            tokens.append(text[i:j])
            i = j
    p = [0]

    def parse():
        if tokens[p[0]] != "(":
            t = tokens[p[0]]
            p[0] += 1
            return t
        p[0] += 1
        out = []
        while tokens[p[0]] != ")":
            out.append(parse())
        p[0] += 1
        return out

    return parse()


def _find(node, tag):
    return [c for c in node if isinstance(c, list) and c and c[0] == tag]


def _first(node, tag):
    for c in node:
        if isinstance(c, list) and c and c[0] == tag:
            return c
    return None


def _unq(s):
    return s[1:-1] if isinstance(s, str) and len(s) >= 2 and s[0] == '"' == s[-1] else s


def _chain_segments(segs, tol=0.01):
    """Walk a soup of (start, end) segments into a single polygon."""
    if not segs:
        return None
    k = lambda p: (round(p[0] / tol) * tol, round(p[1] / tol) * tol)
    rem = list(segs)
    poly = [rem[0][0], rem[0][1]]
    rem.pop(0)
    while rem:
        last = k(poly[-1])
        for i, (a, b) in enumerate(rem):
            if k(a) == last:
                poly.append(b)
                rem.pop(i)
                break
            if k(b) == last:
                poly.append(a)
                rem.pop(i)
                break
        else:
            break
    if len(poly) > 1 and k(poly[0]) == k(poly[-1]):
        poly.pop()
    return np.array(poly)


def import_kicad_pcb(path):
    root = parse_sexp(open(path).read())
    comps = {}

    for fp in _find(root, "footprint"):
        ref = next(
            (_unq(p[2]) for p in _find(fp, "property") if _unq(p[1]) == "Reference"),
            None,
        )
        if not ref:
            continue
        at = _first(fp, "at")
        fx, fy = float(at[1]), -float(at[2])
        ftheta = np.radians(float(at[3])) if len(at) > 3 else 0.0

        pins, xs, ys = [], [], []
        for pad in _find(fp, "pad"):
            pa = _first(pad, "at")
            px, py = float(pa[1]), -float(pa[2])
            nn = _first(pad, "net")
            net = _unq(nn[2]) if nn else f"NC_{ref}_{_unq(pad[1])}"
            pins.append(Pin(np.array([px, py]), net))
            sz = _first(pad, "size")
            sw, sh = float(sz[1]), float(sz[2])
            xs.extend([px - sw / 2, px + sw / 2])
            ys.extend([py - sh / 2, py + sh / 2])

        hw = max(abs(min(xs)), abs(max(xs))) + 0.3 if xs else 1.5
        hh = max(abs(min(ys)), abs(max(ys))) + 0.3 if ys else 1.5
        comps[ref] = Component(
            id=ref,
            pos=np.array([fx, fy]),
            theta=ftheta,
            half_size=np.array([hw, hh]),
            pins=pins,
            allowed_orientations=QUAD,
        )

    segs = []
    for gl in _find(root, "gr_line"):
        ly = _first(gl, "layer")
        if ly and _unq(ly[1]) == "Edge.Cuts":
            s, e = _first(gl, "start"), _first(gl, "end")
            segs.append(((float(s[1]), -float(s[2])), (float(e[1]), -float(e[2]))))
    for gr in _find(root, "gr_rect"):
        ly = _first(gr, "layer")
        if ly and _unq(ly[1]) == "Edge.Cuts":
            s, e = _first(gr, "start"), _first(gr, "end")
            x1, y1 = float(s[1]), -float(s[2])
            x2, y2 = float(e[1]), -float(e[2])
            segs.extend(
                [
                    ((x1, y1), (x2, y1)),
                    ((x2, y1), (x2, y2)),
                    ((x2, y2), (x1, y2)),
                    ((x1, y2), (x1, y1)),
                ]
            )

    board = _chain_segments(segs)
    for c in comps.values():
        c.allowed_polygon = board
    return comps, board


# ---------- Example ----------


def build_example():
    rng = np.random.default_rng(11)
    board = np.array([[0, 0], [30, 0], [30, 20], [15, 20], [15, 10], [0, 10]], float)
    comps = {}

    def add(cid, hs, pin_specs, pos=None, rect=None, theta=0.0, orient=None):
        comps[cid] = Component(
            id=cid,
            pos=np.array(
                pos if pos is not None else rng.uniform([3, 3], [27, 17]), float
            ),
            theta=theta,
            half_size=np.array(hs, float),
            pins=[Pin(np.array(lp, float), net) for lp, net in pin_specs],
            allowed_rect=rect,
            allowed_polygon=board if rect is None else None,
            allowed_orientations=orient if orient is not None else QUAD,
        )

    add(
        "U1",
        (3.0, 3.0),
        [
            ((-2.5, 2.5), "VCC"),
            ((-2.5, -2.5), "GND"),
            ((2.5, 2.5), "USB_DP"),
            ((2.5, 1.5), "USB_DM"),
            ((2.5, 0.5), "SWDIO"),
            ((2.5, -0.5), "SWCLK"),
            ((2.5, -1.5), "LED1_CTRL"),
            ((2.5, -2.5), "LED2_CTRL"),
            ((-2.5, 0.5), "GPIO1"),
            ((-2.5, -0.5), "GPIO2"),
            ((0.5, -2.5), "XTAL1"),
            ((1.5, -2.5), "XTAL2"),
        ],
    )
    add("C1", (0.7, 0.4), [((-0.7, 0), "VCC"), ((0.7, 0), "GND")])
    add("C2", (0.7, 0.4), [((-0.7, 0), "VCC"), ((0.7, 0), "GND")])
    add("R1", (1.0, 0.4), [((-1, 0), "LED1_CTRL"), ((1, 0), "LED1_A")])
    add("D1", (0.8, 0.4), [((-0.8, 0), "LED1_A"), ((0.8, 0), "GND")])
    add("R2", (1.0, 0.4), [((-1, 0), "LED2_CTRL"), ((1, 0), "LED2_A")])
    add("D2", (0.8, 0.4), [((-0.8, 0), "LED2_A"), ((0.8, 0), "GND")])
    add("Y1", (1.4, 0.6), [((-1.4, 0), "XTAL1"), ((1.4, 0), "XTAL2")])

    # Connectors: thin edge strip + single orientation
    add(
        "J_USB",
        (2.5, 1.2),
        [
            ((-1.8, -0.9), "VCC"),
            ((-1.8, -0.3), "USB_DP"),
            ((-1.8, 0.3), "USB_DM"),
            ((-1.8, 0.9), "GND"),
        ],
        pos=(22, 18.5),
        rect=(17.5, 18.5, 28.0, 18.5),
        theta=np.pi / 2,
        orient=[np.pi / 2],
    )
    add(
        "J_DEBUG",
        (2.5, 1.5),
        [
            ((-1.8, -0.9), "SWDIO"),
            ((-1.8, -0.3), "SWCLK"),
            ((-1.8, 0.3), "GND"),
            ((-1.8, 0.9), "VCC"),
        ],
        pos=(28.5, 10),
        rect=(28.5, 3.0, 28.5, 17.0),
        theta=0.0,
        orient=[0.0],
    )
    add(
        "J_GPIO",
        (2.5, 1.2),
        [((-1.8, -0.6), "GPIO1"), ((-1.8, 0.0), "GPIO2"), ((-1.8, 0.6), "GND")],
        pos=(1.5, 5),
        rect=(1.5, 3.0, 1.5, 7.0),
        theta=np.pi,
        orient=[np.pi],
    )
    add(
        "J_PWR",
        (2.0, 1.2),
        [((-1.5, -0.5), "VCC"), ((-1.5, 0.5), "GND")],
        pos=(10, 1.5),
        rect=(4.0, 1.5, 26.0, 1.5),
        theta=3 * np.pi / 2,
        orient=[3 * np.pi / 2],
    )

    return comps, [SwapGroup("R1", [0, 1]), SwapGroup("R2", [0, 1])], board


if __name__ == "__main__":
    import sys

    pcb = (
        sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/outputs/example.kicad_pcb"
    )
    components, board = import_kicad_pcb(pcb)

    # Pick a couple of swap groups by component name (resistors with 2 pins
    # whose endpoints are symmetric and interchangeable).
    swap_groups = [SwapGroup(cid, [0, 1]) for cid in ("R1", "R2") if cid in components]

    # Scramble initial positions so the "before" plot shows the placer doing work.
    # Connectors (J*) are kept at their imported positions and pinned -- typical
    # workflow is to place I/O first, then let the placer arrange everything else.
    rng = np.random.default_rng(0)
    xs, ys = board[:, 0], board[:, 1]
    for c in components.values():
        if c.id.startswith("J"):
            c.fixed = True
            continue
        c.pos = rng.uniform([xs.min() + 2, ys.min() + 2], [xs.max() - 2, ys.max() - 2])
        project_position(c)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw(components, board, axes[0], "Initial")
    run(components, swap_groups, n_iters=800)
    draw(components, board, axes[1], "Converged")

    m = 3
    for ax in axes:
        ax.set_xlim(xs.min() - m, xs.max() + m)
        ax.set_ylim(ys.min() - m, ys.max() + m)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/placement.png", dpi=110)
    print("done")
