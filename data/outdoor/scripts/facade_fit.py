"""Geometry checks for a single outdoor building facade."""
import numpy as np


def fit_facade(boxes, e1, e2, points, heights, frame_ids, cameras, thickness):
    """Fit one storefront plane to verified fixtures, then bound it by multi-view surface support."""
    axes, centers = [], []
    basis2 = np.stack([e1, e2])
    for word, instances in boxes.items():
        if word not in {"door", "poster", "window", "shutter", "metal door", "front door", "roller shutter", "garage door", "shop window"}:
            continue
        for box in instances:
            size = box["尺寸"]
            if size[0] < 0.6 or size[2] < 0.6 or size[1] >= 0.5 * size[0]:
                continue
            axis = basis2 @ box["长轴"]
            axes.append(axis / np.linalg.norm(axis))
            centers.append(basis2 @ box["中心"])
    if len(axes) < 2:
        raise ValueError("单立面模式需要至少两个方向一致的门、窗或海报")
    axes, centers = np.array(axes), np.array(centers)
    # A recessed fixture shifts its center across the facade but preserves its orientation.
    direction = np.linalg.svd(axes, full_matrices=False)[2][0]
    if np.min(np.abs(axes @ direction)) < np.cos(np.radians(12)):
        raise ValueError("立面锚点朝向不一致，请使用多墙模式")
    normal = np.array([-direction[1], direction[0]])
    offsets = centers @ normal
    if np.ptp(offsets) > 0.55 or np.ptp(centers @ direction) < 1:
        raise ValueError("锚点不能确定同一面墙")
    if np.median(cameras @ normal) > offsets.mean():
        normal *= -1
        offsets *= -1
    # Put the visible wall surface just behind the fixtures, so mounting never buries them.
    offset = offsets.max() + thickness / 2
    near = np.abs(points @ normal - offset) <= 0.45
    along, high, frames = points[near] @ direction, heights[near], frame_ids[near]
    step = 0.50
    cells = np.floor(along / step).astype(int)
    support = []
    for cell in np.unique(cells):
        selected = cells == cell
        if np.unique(frames[selected]).size >= 3 and np.ptp(np.percentile(high[selected], [5, 95])) >= 0.5:
            support.append(cell)
    groups = np.split(np.asarray(support, dtype=int), np.where(np.diff(support) != 1)[0] + 1)
    for group in groups:
        if len(group) < 4:
            continue
        lo, hi = group[0] * step, (group[-1] + 1) * step
        if not np.all((centers @ direction >= lo - step) & (centers @ direction <= hi + step)):
            continue
        selected = (along >= lo) & (along <= hi)
        top = np.percentile([np.percentile(high[cells == i], 98) for i in group], 90)
        print(f"[单立面] {len(centers)} 个锚点，{np.unique(frames[selected]).size} 个视角，沿墙 {hi-lo:.2f} m，高 {top:.2f} m")
        return [(direction * lo + normal * offset, direction * hi + normal * offset, float(top))]
    raise ValueError("没有一条得到连续多视角支持的立面覆盖这些锚点")


def facade_explains_walls(facade, walls, supported_lines):
    """Only collapse fragments of one depth band. Keep distinct parallel walls and corners."""
    if not walls:
        return False
    p, q, _ = facade
    length = np.linalg.norm(q - p)
    axis = (q - p) / length
    normal = np.array([-axis[1], axis[0]])
    def alignment(a, b):
        return abs((b - a) @ axis) / np.linalg.norm(b - a)
    # Check before the Manhattan direction filter, which can discard a real return wall.
    if any(alignment(a, b) < np.cos(np.radians(25)) for a, b in supported_lines):
        return False
    for a, b, _ in walls:
        ends = np.stack([a - p, b - p])
        along, off = ends @ axis, ends @ normal
        if along.min() < -1.0 or along.max() > length + 1.0 or np.max(np.abs(off)) > 1.25:
            return False
        if alignment(a, b) > np.cos(np.radians(4)) and off.prod() > 0 and np.min(np.abs(off)) > .6:
            return False
    return True
