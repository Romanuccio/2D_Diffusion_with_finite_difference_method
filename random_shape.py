from PIL import Image, ImageDraw  # For creating and drawing on images
import numpy as np               # For mathematical operations
from numpy.random import default_rng
from scipy.interpolate import splprep, splev  # For spline interpolation

rng = default_rng()

def generate_smooth_blobs(width=128, height=128, num_points=8, noise_factor=0.8, count=None):
    if count is None:
        count = rng.integers(1, 5)
    # Create a new image with white background
    image = Image.new('1', (width, height), color=1)
    draw = ImageDraw.Draw(image)
     
    for i in range(count):
        # Generate initial points in a circle
        center_x, center_y = rng.random() * width, rng.random() * height
        base_radius = np.clip(rng.random() * 10, 5, 15)
        
        # Generate random points around a circle
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        points = []
        for angle in angles:
            # Add random variation to the radius
            radius = base_radius * (1 + noise_factor * (rng.random() - 0.5))
            x = np.clip(center_x + radius * np.cos(angle), 0, width)
            y = np.clip(center_y + radius * np.sin(angle), 0, height)
            points.append([x, y])
        
        # Close the shape by repeating the first point
        points.append(points[0])
        points = np.array(points)
        
        # Fit a B-spline to the points
        tck, _ = splprep([points[:, 0], points[:, 1]], k=3, per=True)
        
        # Generate more points along the spline for smoothness
        u_new = np.linspace(0, 1, 100)
        smooth_points = splev(u_new, tck)
        
        # Convert to list of tuples for PIL
        smooth_points = list(zip(smooth_points[0], smooth_points[1]))
        
        # Draw the shape
        color = 0
        draw.polygon(smooth_points, fill=color)
     
    return image

def generate_smooth_blob(width=128, height=128, num_points=8, noise_factor=0.8):
    # Create a new image with white background
    image = Image.new('1', (width, height), color=1)
    draw = ImageDraw.Draw(image)
     
    # Generate initial points in a circle
    center_x, center_y = rng.random() * width, rng.random() * height
    base_radius = np.clip(rng.random() * 10, 1, 10)
    
    # Generate random points around a circle
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    points = []
    for angle in angles:
        # Add random variation to the radius
        radius = base_radius * (1 + noise_factor * (rng.random() - 0.5))
        x = np.clip(center_x + radius * np.cos(angle), 0, width)
        y = np.clip(center_y + radius * np.sin(angle), 0, height)
        points.append([x, y])
     
    # Close the shape by repeating the first point
    points.append(points[0])
    points = np.array(points)
     
    # Fit a B-spline to the points
    tck, _ = splprep([points[:, 0], points[:, 1]], k=3, per=True)
     
    # Generate more points along the spline for smoothness
    u_new = np.linspace(0, 1, 100)
    smooth_points = splev(u_new, tck)
     
    # Convert to list of tuples for PIL
    smooth_points = list(zip(smooth_points[0], smooth_points[1]))
     
    # Draw the shape
    color = 0
    draw.polygon(smooth_points, fill=color)
     
    return image

def generate_line(width=128, height=128):
    thickness = 10
    A = np.array([width//2, height//2])
    angle = rng.random()*2*np.pi
    B = A + np.array([np.cos(angle), np.sin(angle)])*2*width
    C = A - np.array([np.cos(angle), np.sin(angle)])*2*height
    shift = np.array([rng.integers(-width//2, width//2), rng.integers(-height//2, height//2)])
    B += shift
    C += shift
    pts = [tuple(C), tuple(B)]
    image = Image.new("1", (width, height), color=1)
    draw = ImageDraw.Draw(image)
    draw.line(pts, fill=0, width=thickness)
    return image