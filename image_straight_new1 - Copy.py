import cv2
import numpy as np
import os
import os
import json

# Function to transform a part of the image
def transform_part(image, src_part_points):

    src_points_np = np.array(src_part_points, dtype=np.float32)
    p0, p1, p2, p3 = src_points_np
    width = int(max(np.linalg.norm(p0 - p1), np.linalg.norm(p2 - p3)))
    height = int(max(np.linalg.norm(p0 - p2), np.linalg.norm(p1 - p3)))

    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points_np, dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped

# Function to straighten the image
def straighten_image(image, src_points):
    if len(src_points) != 6:
        raise ValueError("Exactly 6 points required.")

    top_left, mid_up, top_right, bottom_left, mid_bottom, bottom_right = src_points
    # Transform left part
    left_part = transform_part(image, [top_left, mid_up, bottom_left, mid_bottom])
    
    # Transform right part
    right_part = transform_part(image, [mid_up, top_right, mid_bottom, bottom_right])

    common_height = int(np.linalg.norm(np.array(mid_up) - np.array(mid_bottom)))

    # Resize both parts to the common height if needed
    if left_part.shape[0] != common_height:
        left_part = cv2.resize(left_part, (left_part.shape[1], common_height))
    if right_part.shape[0] != common_height:
        right_part = cv2.resize(right_part, (right_part.shape[1], common_height))

    # Merge the two parts
    # merged_height = max(left_part.shape[0], right_part.shape[0]) 
    merged_width = left_part.shape[1] + right_part.shape[1]
    merged_image = np.zeros((common_height, merged_width, 3), dtype=np.uint8)

    merged_image[:left_part.shape[0], :left_part.shape[1]] = left_part
    merged_image[:right_part.shape[0], left_part.shape[1]:] = right_part

    return merged_image


def parse_json(data):
    x=data.get("image")    
    a=x[0]
    b=x[1]
    c=x[2]
    d=x[3]
    e=x[4]
    f=x[5]
    upper_left=(a["clickX"],a["clickY"])
    upper_mid=(b["clickX"],b["clickY"])
    upper_right=(c["clickX"],c["clickY"])
    lower_left=(d["clickX"],d["clickY"])
    lower_mid=(e["clickX"],e["clickY"])
    lower_right=(f["clickX"],f["clickY"])    
    return upper_left,upper_mid,upper_right,lower_left,lower_mid,lower_right

def arrange_points(points):
    # Extract x-values to determine boundaries for left, mid, and right
    x_values = [x for x, _ in points]
    min_x, max_x = min(x_values), max(x_values)

    # Define ranges based on x-values
    range_1 = min_x + (max_x - min_x) / 3  # Left boundary
    range_2 = min_x + 2 * (max_x - min_x) / 3  # Right boundary

    # Split upper_part into left, mid, and right based on x-value
    left = [point for point in points if point[0] < range_1]
    mid = [point for point in points if range_1 <= point[0] < range_2]
    right = [point for point in points if point[0] >= range_2]

    return left, mid,right

def find_upper_lower_point(data):
    
    y_values = [y for _, y in data]
    threshold_y = sum(y_values) / len(y_values)  # Mean of y-values

    # Split into upper_part and lower_part based on y-value
    upper_part = [point for point in data if point[1] < threshold_y]
    lower_part = [point for point in data if point[1] >= threshold_y]
    return upper_part ,lower_part


if __name__=="__main__":

    dir=r"C:\Users\ASHUTOSH\Desktop\New folder"#(json folder) C:\Users\ASHUTOSH\Desktop\out
    image_folder = r"C:\Users\ASHUTOSH\Desktop\error"#(image folder)
    #output_folder = r'C:\Users\ASHUTOSH\Desktop\out\out'
    college_name_folder=os.path.join(image_folder,"college_name_area")
    os.makedirs(college_name_folder, exist_ok=True)
    tr_data_folder=os.path.join(image_folder,"Tr_data_area")
    os.makedirs(tr_data_folder, exist_ok=True)

    json_dict={}

    for filename in os.listdir(dir):
        file=filename.split(".")[0]
        file_path=os.path.join(dir,filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
       
        # with open(f'{file_path}.json', 'r', encoding='ISO-8859-1') as f:
            data = json.load(f)
            json_dict[file]=data
            

    filename_list = [filename for filename in os.listdir(image_folder) if filename.lower().endswith(('.jpg','.jpeg','.png'))]
    print("filename_list ",filename_list)
    for i, filename in enumerate(filename_list):
        image_path=os.path.join( image_folder,filename)
        image=cv2.imread(image_path)
        org_height,org_width= image.shape[:2]
        file=filename.split(".")[0]
        data= json_dict[file]
        upper_left,upper_mid,upper_right,lower_left,lower_mid,lower_right=parse_json(data)
        src_points_org=[upper_left,upper_mid,upper_right,lower_left,lower_mid,lower_right]

        upper_part ,lower_part=find_upper_lower_point(src_points_org)
        upper_left,upper_mid, upper_right=arrange_points(upper_part)
        upper_left = upper_left[0]
        upper_mid = upper_mid[0]
        upper_right = upper_right[0]
        
        lower_left,lower_mid, lower_right=arrange_points(lower_part)
        lower_left = lower_left[0]
        lower_mid = lower_mid[0]
        lower_right = lower_right[0]
        
        src_points=[upper_left,upper_mid,upper_right,lower_left,lower_mid,lower_right]
        mapped_src_points = [(int(src_point[0])*5 , int(src_point[1])*5) for src_point in src_points]
        crop_dimension=mapped_src_points[0]
        crop_x=crop_dimension[0]
        crop_y=crop_dimension[1]

        cropped_upper_portion_image=image[:crop_y-30,:]
        crop_path = os.path.join(college_name_folder, filename)
        cv2.imwrite(crop_path,cropped_upper_portion_image )

        print("crop_x,crop_y",crop_x,crop_y)
        print(f"src_points for {filename} \n",src_points)
        print(f'mapped_src_points for {filename} \n', mapped_src_points)

        if image is None:
            raise ValueError("Could not load image. Check the file path.")
        
        transformed = straighten_image(image, mapped_src_points)
        output_path = os.path.join(tr_data_folder, filename)

        cv2.imwrite(output_path, transformed)
 