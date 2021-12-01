def intersection_over_union(bbox1, bbox2) -> float:
    area_intersection = intersection_area(bbox1, bbox2)
    area_union = union_area(bbox1, bbox2)

    return area_intersection / area_union if area_union != 0 else 0


def union_area(bbox1, bbox2) -> float:
    x1_l, y1_h, x1_h, y1_l = bbox1
    x2_l, y2_h, x2_h, y2_l = bbox2

    bbox1_area = (x1_h - x1_l) * (y1_h - y1_l)
    bbox2_area = (x2_h - x2_l) * (y2_h - y2_l)

    area_intersection = intersection_area(bbox1, bbox2)

    return bbox1_area + bbox2_area - area_intersection


def intersection_area(bbox1, bbox2) -> float:
    x1_l, y1_h, x1_h, y1_l = bbox1
    x2_l, y2_h, x2_h, y2_l = bbox2

    if x1_h < x2_l or x2_h < x1_l:
        return 0.0
    if y1_h < y2_l or y2_h < y2_l:
        return 0.0

    h_pts = [
        min(x1_l, x2_l), max(x1_l, x2_l), min(x1_h, x2_h), max(x1_h, x2_h)
    ]
    v_pts = [
        min(y1_l, y2_l), max(y1_l, y2_l), min(y1_h, y2_h), max(y1_h, y2_h)
    ]

    return (h_pts[2] - h_pts[1]) * (v_pts[2] - v_pts[1])


def iou_test_driver():
    bbox1 = (1, 10, 6, 3)
    bbox2 = (3, 15, 8, 7)
    iou = intersection_over_union(bbox1, bbox2)
    intersection = intersection_area(bbox1, bbox2)
    union = union_area(bbox1, bbox2)
    print(f"For {bbox1} and {bbox2}: \n"
          f"Intersection area is: {intersection}\n"
          f"Union area is: {union}\n"
          f"Intersection over union is: {iou}\n")


if __name__ == '__main__':
    iou_test_driver()
