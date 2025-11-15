import cv2
import numpy as np

def detect_red_corners(image):
    """
    Detect the 4 red corner squares and return their centroids
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color in HSV
    # Red wraps around in HSV, so we need two ranges
    # More permissive ranges to catch different red shades and lighting
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of red regions
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} red regions")
    
    # Find centroids of red squares
    red_regions = []
    h, w = image.shape[:2]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Store all red regions with decent size
            if area > 200:  # Lower threshold to detect smaller red markers
                red_regions.append((cx, cy, area))
                print(f"  Red region at ({cx}, {cy}), area={area:.0f}")
    
    # Now identify the 4 corners from the detected red regions
    # The corners should be the 4 most extreme points
    if len(red_regions) < 4:
        print(f"⚠️ Only found {len(red_regions)} sizable red regions")
        return [], red_mask
    
    # Convert to numpy array for easier manipulation
    points = np.array([(x, y) for x, y, _ in red_regions])
    
    corners = []
    
    # Find the 4 extreme corners
    # Top-left: minimum x+y
    sums = points[:, 0] + points[:, 1]
    top_left_idx = np.argmin(sums)
    corners.append(tuple(points[top_left_idx]))
    
    # Bottom-right: maximum x+y
    bottom_right_idx = np.argmax(sums)
    corners.append(tuple(points[bottom_right_idx]))
    
    # Top-right: maximum x-y
    diffs = points[:, 0] - points[:, 1]
    top_right_idx = np.argmax(diffs)
    corners.append(tuple(points[top_right_idx]))
    
    # Bottom-left: minimum x-y
    bottom_left_idx = np.argmin(diffs)
    corners.append(tuple(points[bottom_left_idx]))
    
    # Remove duplicates and get exactly 4 unique corners
    corners = list(set(corners))
    
    if len(corners) != 4:
        print(f"⚠️ Found {len(corners)} unique corners instead of 4")
        # Try alternative method: find 4 points closest to actual image corners
        image_corners = [
            (0, 0),           # Top-left
            (w, 0),           # Top-right
            (w, h),           # Bottom-right
            (0, h)            # Bottom-left
        ]
        
        corners = []
        for img_corner in image_corners:
            # Find closest red region to each image corner
            distances = [np.sqrt((p[0]-img_corner[0])**2 + (p[1]-img_corner[1])**2) 
                        for p in points]
            closest_idx = np.argmin(distances)
            corners.append(tuple(points[closest_idx]))
    
    # Order the corners correctly
    corners = order_corner_points(corners)
    
    return corners, red_mask
    
    # Sort corners by area and take the 4 largest (most likely to be corner markers)
    red_corners.sort(key=lambda x: x[2], reverse=True)
    red_corners = red_corners[:4]
    
    # Remove area information and keep only coordinates
    corner_points = [(x, y) for x, y, _ in red_corners]
    
    # Order the corners correctly: top-left, top-right, bottom-right, bottom-left
    corner_points = order_corner_points(corner_points)
    
    return corner_points, red_mask


def order_corner_points(points):
    """
    Order points as: top-left, top-right, bottom-right, bottom-left
    """
    if len(points) != 4:
        return points
    
    # Convert to numpy array
    pts = np.array(points)
    
    # Find the center point
    center = np.mean(pts, axis=0)
    
    # Sort points by angle from center
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Create ordered array
    ordered = np.zeros((4, 2), dtype=np.float32)
    
    # Sum and diff to identify corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    
    # Top-left has smallest sum
    ordered[0] = pts[np.argmin(s)]
    # Bottom-right has largest sum
    ordered[2] = pts[np.argmax(s)]
    # Top-right has smallest difference
    ordered[1] = pts[np.argmin(diff)]
    # Bottom-left has largest difference
    ordered[3] = pts[np.argmax(diff)]
    
    return ordered


def transform_arena_auto(input_path, output_path, world_corners=None, output_width=960, output_height=720):
    """
    Automatically detect red corners and perform perspective transformation
    
    Args:
        input_path: Path to input image
        output_path: Path to save transformed image
        world_corners: If provided, these are the real-world coordinates of the corners
        output_width: Width of output image
        output_height: Height of output image
    """
    
    # Read image
    img = cv2.imread(input_path)
    
    print("Detecting red corner markers...")
    
    # Detect red corners
    corner_points, red_mask = detect_red_corners(img)
    
    if len(corner_points) != 4:
        print(f"⚠️ Warning: Found {len(corner_points)} corners instead of 4")
        print("Trying manual fallback or adjusting parameters...")
        return None, None
    
    print(f"✅ Found 4 corner markers!")
    print("\nDetected corner coordinates (pixels):")
    labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
    for i, (x, y) in enumerate(corner_points):
        print(f"  {labels[i]}: ({x:.0f}, {y:.0f})")
    
    # Convert to float32
    src_points = np.float32(corner_points)
    
    # Define destination points (perfect rectangle)
    dst_points = np.float32([
        [0, 0],                      # Top-left
        [output_width, 0],           # Top-right
        [output_width, output_height], # Bottom-right
        [0, output_height]           # Bottom-left
    ])
    
    # Calculate perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    transformed = cv2.warpPerspective(img, matrix, (output_width, output_height))
    
    # Save output
    cv2.imwrite(output_path, transformed)
    print(f"\n✅ Transformed image saved to: {output_path}")
    
    # Save the transformation matrix
    np.save('auto_transform_matrix.npy', matrix)
    print("✅ Transformation matrix saved")
    
    # Create visualization showing detected corners
    vis_img = img.copy()
    for i, (x, y) in enumerate(corner_points):
        # Draw circle at corner
        cv2.circle(vis_img, (int(x), int(y)), 10, (0, 255, 0), -1)
        # Add label
        cv2.putText(vis_img, str(i+1), (int(x)+15, int(y)-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw the quadrilateral
    pts = np.int32(corner_points)
    cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3)
    
    # Save visualization
    cv2.imwrite('detected_corners.png', vis_img)
    cv2.imwrite('red_mask.png', red_mask)
    print("✅ Corner detection visualization saved")
    
    # If world coordinates are provided, calculate real-world transform
    if world_corners is not None:
        print("\n" + "="*60)
        print("WORLD COORDINATE CALIBRATION")
        print("="*60)
        world_points = np.float32(world_corners)
        world_matrix, _ = cv2.findHomography(src_points, world_points)
        np.save('world_transform_matrix.npy', world_matrix)
        print("✅ World coordinate transformation matrix saved")
        
        print("\nWorld coordinates of corners (meters):")
        print("  Bottom-left (ORIGIN) → Top-left → Top-right → Bottom-right")
        for i, (x, y) in enumerate(world_corners):
            corner_name = ["Top-left", "Top-right (FAR)", "Bottom-right", "Bottom-left (ORIGIN)"][i]
            print(f"  {corner_name}: ({x:.2f}m, {y:.2f}m)")
    
    return transformed, matrix


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    # Input and output paths
    INPUT_IMAGE = 'test-real.png'
    OUTPUT_IMAGE = 'arena_transformed_auto.png'
    
    # OPTIONAL: Define real-world coordinates of the red corners (in meters)
    # Using your actual measured coordinates
    # Note: The arena appears slightly skewed (not a perfect rectangle)
    WORLD_CORNERS = [
        (0, 3.85),      # Top-left corner in meters
        (2.35, 3.95),   # Top-right corner in meters
        (1.7, 0.05),    # Bottom-right corner in meters
        (0.45, 0)       # Bottom-left corner in meters (closest to origin)
    ]
    
    # Output image dimensions
    # Swapping to correct order: WIDTH x HEIGHT
    OUTPUT_HEIGHT = 1280   # height in pixels
    OUTPUT_WIDTH = 720  # width in pixels
    
    # Run the automatic transformation
    print("="*60)
    print("AUTOMATIC RED CORNER DETECTION AND TRANSFORMATION")
    print("="*60)
    
    transformed, matrix = transform_arena_auto(
        INPUT_IMAGE,
        OUTPUT_IMAGE,
        world_corners=WORLD_CORNERS,  # Optional: pass None if you don't need world coordinates
        output_width=OUTPUT_WIDTH,
        output_height=OUTPUT_HEIGHT
    )
    
    if transformed is not None:
        print("\n" + "="*60)
        print("SUCCESS! Your arena has been transformed.")
        print("="*60)
        print("\nOutput files:")
        print("  - Transformed arena: arena_transformed_auto.png")
        print("  - Detected corners: detected_corners.png")
        print("  - Red detection mask: red_mask.png")
        print("  - Transform matrix: auto_transform_matrix.npy")
        
        print("\nYou can now use this transformation matrix on any image from the same camera position!")
    else:
        print("\n⚠️ Transformation failed. Please check:")
        print("  1. All 4 red corner markers are clearly visible")
        print("  2. No other large red objects are in the corners")
        print("  3. Lighting conditions allow red detection")