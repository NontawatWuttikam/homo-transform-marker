import cv2
import os
import numpy as np
import copy

# vid_path = "test3.mp4"

cap = cv2.VideoCapture(0)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 10  # Minimum area in pixels

params.filterByCircularity = True
params.minCircularity = 0.2

params.filterByConvexity = True
params.minConvexity = 0.2

params.filterByInertia = True
params.minInertiaRatio = 0.2

detector = cv2.SimpleBlobDetector_create(params)

def sort_points(points):
    centroid = np.mean(points, axis=0)

    left_points = []
    right_points = []

    for point in points:
        if point[0] < centroid[0]:
            left_points.append(point)
        else: 
            right_points.append(point)

    if len(left_points) < 2 or len(right_points) < 2: return False #

    left_points = sorted(left_points, key=lambda p: p[1])
    left_top, left_bottom = left_points

    right_points = sorted(right_points, key=lambda p: p[1], reverse=True)
    right_bottom, right_top = right_points

    sorted_points = [left_top, left_bottom, right_bottom, right_top]

    return np.array(sorted_points)


imgsz_factor = 0.7
map_space = np.full((1000, 1000, 3), 20, dtype="uint8")
map_points = np.array([[0,0], [0,map_space.shape[0]], [map_space.shape[1], map_space.shape[0]], [map_space.shape[1], 0]])

for map_point in map_points:
    map_space = cv2.circle(map_space, map_point.astype(np.int16), 50, (0,255,0), -1)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while True:
        ret, frame = cap.read()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hue = frame_hsv[:,:,0]
        sat = frame_hsv[:,:,1]
        val = frame_hsv[:,:,2]

        green_mask = (hue > 50) & (hue < 70) & (sat > 60)

        yellow_mask = (hue > 100) & (hue < 120) & (sat > 80)

        yellow = (yellow_mask[:,:,None]* 255).astype(np.uint8)
        green = (green_mask[:,:,None]* 255).astype(np.uint8)

        kernel_size = 5

        green = cv2.erode(green, np.ones((kernel_size,kernel_size)))
        yellow = cv2.erode(yellow, np.ones((kernel_size,kernel_size)))

        zeros = np.zeros(green_mask.shape, np.uint8)[:,:,None]
        green_visualize = np.dstack((zeros, green, zeros))
        yellow_visualize = np.dstack((yellow, zeros , zeros))

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(green)
        if nb_components == 1: continue

        stats = stats[1:]
        sort_stat = stats[:,4].argsort()
        green_positions = centroids[1:][sort_stat][:4]

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(yellow)
        if nb_components == 1: continue

        stats = stats[1:]
        # max_id = stats[:,4].argmax()
        centroids = centroids[1:]
        green_avg = green_positions.mean(axis=0)
        print(green_avg)
        closest_yellow = centroids[0]
        closest_dist = np.linalg.norm(closest_yellow - green_avg)
        for centroid in centroids:
            dist = np.linalg.norm(centroid - green_avg)
            if dist < closest_dist:
                closest_dist = dist
                closest_yellow = centroid
        yellow_position = centroid
        # print(green_positions,stats)

        if not ret:
            print("End of video or can't retrieve the frame.")
            break


        # Draw detected blobs as red circles on the original image
        viz = green_visualize + yellow_visualize


        if len(green_positions) == 4:
            green_positions = sort_points(green_positions)
            if type(green_positions) != np.ndarray: continue
            M, mask = cv2.findHomography(green_positions, map_points, cv2.RANSAC,5.0)
            homogenized_yellow = np.concatenate((yellow_position, [1]))
            yt = M @ homogenized_yellow
            yt = np.array([yt[0]/yt[2], yt[1]/yt[2]])

            drawed_map = cv2.circle(copy.deepcopy(map_space), yt.round().astype(np.int16), 20, (0,255,255), -1)
            print("estimated M",M)
        
            for green_position in green_positions:
                viz = cv2.circle(viz, green_position.astype(np.int16), 10, (0,128,0), -1)
            viz = cv2.circle(viz, yellow_position.astype(np.int16), 10, (128,0,0), -1)

            drawed_map = cv2.resize(drawed_map, (int(round(drawed_map.shape[1]*imgsz_factor)), int(round(drawed_map.shape[0]*imgsz_factor))))
            frame = cv2.resize(frame, (int(round(frame.shape[1]*imgsz_factor)), int(round(frame.shape[0]*imgsz_factor))))
            viz = cv2.resize(viz, (int(round(viz.shape[1]*imgsz_factor)), int(round(viz.shape[0]*imgsz_factor))))

            cv2.imshow('Image Space', frame)
            cv2.imshow('Detection', viz)
            cv2.imshow('Map space', drawed_map)

        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
