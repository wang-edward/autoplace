"""
Force-directed PCB autoplacer v2.

New in this version:
  - Arbitrary polygon board outline (point-in-polygon + closest-point projection)
  - All components restricted to 90-degree orientations
  - Connectors with a single fixed orientation + thin edge strip for position
  - Example board is L-shaped to exercise the non-convex polygon projection

Algorithm structure unchanged: overdamped descent + AABB repulsion +
Hungarian-based pin swap + annealed orientation snap.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches


QUAD = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]


# ---------- Data model ----------


@dataclass
class Pin:
    name: str
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
    allowed_rect: tuple | None = None  # (xmin, ymin, xmax, ymax)
    allowed_polygon: np.ndarray | None = None  # Nx2 CCW polygon
    allowed_orientations: list | None = None  # list of angles in radians


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


def point_in_polygon(p, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > p[1]) != (yj > p[1])) and (
            p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def closest_on_segment(p, a, b):
    ab = b - a
    denom = float(ab @ ab) + 1e-12
    t = max(0.0, min(1.0, float((p - a) @ ab) / denom))
    return a + t * ab


def closest_on_polygon(p, poly):
    poly = np.asarray(poly, float)
    best = None
    best_d = np.inf
    for i in range(len(poly)):
        q = closest_on_segment(p, poly[i], poly[(i + 1) % len(poly)])
        d = float((p - q) @ (p - q))
        if d < best_d:
            best_d = d
            best = q
    return best


# ---------- Forces ----------


def attractive(components, nets, k=1.0):
    F_lin = {cid: np.zeros(2) for cid in components}
    F_tor = {cid: 0.0 for cid in components}
    for net, members in nets.items():
        if len(members) < 2:
            continue
        positions = [
            pin_world(components[cid], components[cid].pins[pi]) for cid, pi in members
        ]
        center = np.mean(positions, axis=0)
        for (cid, pi), pos in zip(members, positions):
            f = k * (center - pos)
            F_lin[cid] += f
            r = pos - components[cid].pos
            F_tor[cid] += r[0] * f[1] - r[1] * f[0]
    return F_lin, F_tor


def repulsive(components, k=30.0, slop=0.4):
    F_lin = {cid: np.zeros(2) for cid in components}
    ids = list(components.keys())
    for i in range(len(ids)):
        a = components[ids[i]]
        for j in range(i + 1, len(ids)):
            b = components[ids[j]]
            # Use *rotated* extent on each axis. For 90-deg rotations only,
            # half_size swaps when theta is ~pi/2 or 3pi/2.
            ah = rotated_half_size(a)
            bh = rotated_half_size(b)
            dx = b.pos[0] - a.pos[0]
            dy = b.pos[1] - a.pos[1]
            ox = (ah[0] + bh[0] + slop) - abs(dx)
            oy = (ah[1] + bh[1] + slop) - abs(dy)
            if ox > 0 and oy > 0:
                if ox < oy:
                    push = np.array([np.sign(dx if dx != 0 else 1) * ox * k, 0.0])
                else:
                    push = np.array([0.0, np.sign(dy if dy != 0 else 1) * oy * k])
                F_lin[ids[i]] -= push
                F_lin[ids[j]] += push
    return F_lin


def rotated_half_size(comp):
    """Half-size after accounting for 90-degree rotations."""
    # For arbitrary theta this isn't exact, but for {0, pi/2, pi, 3pi/2}
    # (which we enforce) the AABB just swaps axes at odd multiples.
    snapped = round(comp.theta / (np.pi / 2)) % 2
    if snapped == 0:
        return comp.half_size
    return np.array([comp.half_size[1], comp.half_size[0]])


# ---------- Constraints ----------


def project_body_into_polygon(comp, polygon, max_iters=6):
    """Push center until all 4 corners of the (axis-aligned) body lie inside.
    Assumes orientation is a multiple of 90 degrees, so the body is AABB-shaped."""
    for _ in range(max_iters):
        hs = rotated_half_size(comp)
        corners = comp.pos + np.array(
            [
                [-hs[0], -hs[1]],
                [hs[0], -hs[1]],
                [hs[0], hs[1]],
                [-hs[0], hs[1]],
            ]
        )
        push = np.zeros(2)
        any_out = False
        for corner in corners:
            if point_in_polygon(corner, polygon):
                continue
            q = closest_on_polygon(corner, polygon)
            d = q - corner
            # max magnitude per axis across violating corners
            if abs(d[0]) > abs(push[0]):
                push[0] = d[0]
            if abs(d[1]) > abs(push[1]):
                push[1] = d[1]
            any_out = True
        if not any_out:
            return
        comp.pos = comp.pos + push


def project_position(comp):
    if comp.allowed_rect is not None:
        x0, y0, x1, y1 = comp.allowed_rect
        comp.pos[0] = np.clip(comp.pos[0], x0, x1)
        comp.pos[1] = np.clip(comp.pos[1], y0, y1)
        return
    if comp.allowed_polygon is not None:
        if not point_in_polygon(comp.pos, comp.allowed_polygon):
            comp.pos = closest_on_polygon(comp.pos, comp.allowed_polygon)
        project_body_into_polygon(comp, comp.allowed_polygon)


def snap_orientation(comp, strength):
    if comp.allowed_orientations is None or strength <= 0:
        return
    diffs = [
        ((a - comp.theta + np.pi) % (2 * np.pi) - np.pi, a)
        for a in comp.allowed_orientations
    ]
    delta, _ = min(diffs, key=lambda d: abs(d[0]))
    comp.theta += strength * delta


def hard_snap_orientation(comp):
    if comp.allowed_orientations is None:
        return
    diffs = [
        ((a - comp.theta + np.pi) % (2 * np.pi) - np.pi, a)
        for a in comp.allowed_orientations
    ]
    _, target = min(diffs, key=lambda d: abs(d[0]))
    comp.theta = target


# ---------- Pin swap (Hungarian) ----------


def optimize_swap_group(components, nets, group):
    comp = components[group.component_id]
    pins = [comp.pins[i] for i in group.pin_indices]
    nets_in_group = [p.net for p in pins]
    pin_positions = [pin_world(comp, p) for p in pins]

    centroids = []
    for net in nets_in_group:
        positions = []
        for cid, pi in nets[net]:
            if cid == comp.id and pi in group.pin_indices:
                continue
            c = components[cid]
            positions.append(pin_world(c, c.pins[pi]))
        centroids.append(np.mean(positions, axis=0) if positions else comp.pos)

    n = len(group.pin_indices)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.sum((pin_positions[i] - centroids[j]) ** 2)

    row, col = linear_sum_assignment(cost)
    new_nets = [None] * n
    for r, c in zip(row, col):
        new_nets[r] = nets_in_group[c]
    for slot, new_net in zip(group.pin_indices, new_nets):
        comp.pins[slot].net = new_net


def rebuild_nets(components):
    nets = {}
    for cid, comp in components.items():
        for pi, pin in enumerate(comp.pins):
            nets.setdefault(pin.net, []).append((cid, pi))
    return nets


# ---------- Main loop ----------


def run(components, swap_groups, n_iters=800, dt0=0.05, swap_every=25, snap_phase=0.4):
    dt = dt0
    for it in range(n_iters):
        nets = rebuild_nets(components)
        Fa_lin, Fa_tor = attractive(components, nets)
        Fr_lin = repulsive(components)

        progress = it / n_iters
        snap = max(0.0, (progress - snap_phase) / (1 - snap_phase)) ** 2

        for cid, comp in components.items():
            if comp.fixed:
                continue
            f = Fa_lin[cid] + Fr_lin[cid]
            comp.pos = comp.pos + dt * f

            # Freeze rotation if exactly one orientation is allowed.
            single_orient = (
                comp.allowed_orientations is not None
                and len(comp.allowed_orientations) == 1
            )
            if single_orient:
                comp.theta = comp.allowed_orientations[0]
            else:
                comp.theta = comp.theta + dt * 0.15 * Fa_tor[cid]
                snap_orientation(comp, snap * 0.5)

            project_position(comp)

        if it > 0 and it % swap_every == 0:
            for g in swap_groups:
                optimize_swap_group(components, nets, g)

        dt *= 0.996

    # Final hard snap for orientation, then one final swap pass.
    for cid, comp in components.items():
        if not comp.fixed:
            hard_snap_orientation(comp)
            project_position(comp)
    nets = rebuild_nets(components)
    for g in swap_groups:
        optimize_swap_group(components, nets, g)


# ---------- Visualization ----------


def draw(components, board_polygon, ax, title=""):
    nets = rebuild_nets(components)
    ax.set_aspect("equal")
    ax.set_title(title)

    # Board outline
    ax.add_patch(
        patches.Polygon(
            board_polygon,
            fill=True,
            facecolor="#f6f6e8",
            edgecolor="#444",
            linewidth=1.6,
            zorder=0,
        )
    )

    # Net stars
    for net, members in nets.items():
        if net in ("GND", "VCC"):
            continue
        pts = [
            pin_world(components[cid], components[cid].pins[pi]) for cid, pi in members
        ]
        if len(pts) < 2:
            continue
        center = np.mean(pts, axis=0)
        for p in pts:
            ax.plot(
                [p[0], center[0]],
                [p[1], center[1]],
                "-",
                color="#6b6",
                alpha=0.5,
                linewidth=0.8,
            )

    # Components
    for cid, c in components.items():
        R = rot(c.theta)
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * c.half_size
        corners = (R @ corners.T).T + c.pos
        is_conn = cid.startswith("J_")
        color = "#fcc" if is_conn else "#cce"
        ax.add_patch(
            patches.Polygon(
                corners,
                fill=True,
                facecolor=color,
                edgecolor="#225",
                linewidth=1.0,
                zorder=2,
            )
        )
        ax.text(
            c.pos[0], c.pos[1], cid, ha="center", va="center", fontsize=6.5, zorder=3
        )
        # Pins
        for p in c.pins:
            wp = pin_world(c, p)
            ax.plot(wp[0], wp[1], "o", color="#225", markersize=1.8, zorder=3)


# ---------- Example ----------


def build_example():
    rng = np.random.default_rng(11)
    components = {}

    # L-shaped board, CCW
    board = np.array([[0, 0], [30, 0], [30, 20], [15, 20], [15, 10], [0, 10]], float)

    def add(cid, hs, pins, pos=None, **kw):
        components[cid] = Component(
            id=cid,
            pos=np.array(
                pos if pos is not None else rng.uniform([3, 3], [27, 17]), float
            ),
            half_size=np.array(hs, float),
            pins=[Pin(n, np.array(lp, float), net) for n, lp, net in pins],
            **kw,
        )

    # MCU
    add(
        "U1",
        (3.0, 3.0),
        [
            ("VDD", (-2.5, 2.5), "VCC"),
            ("VSS", (-2.5, -2.5), "GND"),
            ("USB_DP", (2.5, 2.5), "USB_DP"),
            ("USB_DM", (2.5, 1.5), "USB_DM"),
            ("SWDIO", (2.5, 0.5), "SWDIO"),
            ("SWCLK", (2.5, -0.5), "SWCLK"),
            ("LED1", (2.5, -1.5), "LED1_CTRL"),
            ("LED2", (2.5, -2.5), "LED2_CTRL"),
            ("GPIO1", (-2.5, 0.5), "GPIO1"),
            ("GPIO2", (-2.5, -0.5), "GPIO2"),
            ("XTAL1", (0.5, -2.5), "XTAL1"),
            ("XTAL2", (1.5, -2.5), "XTAL2"),
        ],
    )

    # Decoupling
    add("C1", (0.7, 0.4), [("1", (-0.7, 0), "VCC"), ("2", (0.7, 0), "GND")])
    add("C2", (0.7, 0.4), [("1", (-0.7, 0), "VCC"), ("2", (0.7, 0), "GND")])

    # LED chains (R has swap group)
    add("R1", (1.0, 0.4), [("1", (-1, 0), "LED1_CTRL"), ("2", (1, 0), "LED1_A")])
    add("D1", (0.8, 0.4), [("A", (-0.8, 0), "LED1_A"), ("K", (0.8, 0), "GND")])
    add("R2", (1.0, 0.4), [("1", (-1, 0), "LED2_CTRL"), ("2", (1, 0), "LED2_A")])
    add("D2", (0.8, 0.4), [("A", (-0.8, 0), "LED2_A"), ("K", (0.8, 0), "GND")])

    # Crystal
    add("Y1", (1.4, 0.6), [("1", (-1.4, 0), "XTAL1"), ("2", (1.4, 0), "XTAL2")])

    # ----- Connectors (edge-constrained, single orientation) -----

    def make_conn(cid, hs, pins, rect, theta, init_pos):
        add(cid, hs, pins, pos=init_pos)
        c = components[cid]
        c.allowed_rect = rect
        c.allowed_orientations = [theta]
        c.theta = theta

    # J_USB: top edge of right column (y = 18.5), faces up
    make_conn(
        "J_USB",
        (2.5, 1.2),
        [
            ("VBUS", (-1.8, -0.9), "VCC"),
            ("DP", (-1.8, -0.3), "USB_DP"),
            ("DM", (-1.8, 0.3), "USB_DM"),
            ("GND", (-1.8, 0.9), "GND"),
        ],
        rect=(17.5, 18.5, 28.0, 18.5),
        theta=np.pi / 2,
        init_pos=(22.0, 18.5),
    )

    # J_DEBUG: right edge (x = 28.5), faces right
    make_conn(
        "J_DEBUG",
        (2.5, 1.5),
        [
            ("SWDIO", (-1.8, -0.9), "SWDIO"),
            ("SWCLK", (-1.8, -0.3), "SWCLK"),
            ("GND", (-1.8, 0.3), "GND"),
            ("VCC", (-1.8, 0.9), "VCC"),
        ],
        rect=(28.5, 3.0, 28.5, 17.0),
        theta=0.0,
        init_pos=(28.5, 10.0),
    )

    # J_GPIO: left edge of bottom strip (x = 1.5), faces left
    make_conn(
        "J_GPIO",
        (2.5, 1.2),
        [
            ("G1", (-1.8, -0.6), "GPIO1"),
            ("G2", (-1.8, 0.0), "GPIO2"),
            ("GND", (-1.8, 0.6), "GND"),
        ],
        rect=(1.5, 3.0, 1.5, 7.0),
        theta=np.pi,
        init_pos=(1.5, 5.0),
    )

    # J_PWR: bottom edge (y = 1.5), faces down
    make_conn(
        "J_PWR",
        (2.0, 1.2),
        [("V+", (-1.5, -0.5), "VCC"), ("GND", (-1.5, 0.5), "GND")],
        rect=(4.0, 1.5, 26.0, 1.5),
        theta=3 * np.pi / 2,
        init_pos=(10.0, 1.5),
    )

    # Apply board polygon + 90-deg orientations to all non-connectors
    for cid, c in components.items():
        if cid.startswith("J_"):
            continue
        c.allowed_polygon = board
        c.allowed_orientations = QUAD

    swap_groups = [SwapGroup("R1", [0, 1]), SwapGroup("R2", [0, 1])]
    return components, swap_groups, board


if __name__ == "__main__":
    components, swap_groups, board = build_example()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw(components, board, axes[0], "Initial")

    run(components, swap_groups, n_iters=800)

    draw(components, board, axes[1], "Converged")
    for ax in axes:
        ax.set_xlim(-2, 32)
        ax.set_ylim(-2, 22)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("result.png", dpi=110)
    print("done")
