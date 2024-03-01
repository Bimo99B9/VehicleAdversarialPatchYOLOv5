import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageOps, ImageDraw, ImageFont


# Function to detect objects using YOLO
def detect_objects(image, yolo_model):
    transform = transforms.Compose([transforms.ToTensor()])
    resized_padded_image = resize_and_pad(image)
    img_tensor = transform(resized_padded_image).unsqueeze(0)
    results = yolo_model(img_tensor)
    return results[0], resized_padded_image.size


# Function to draw bounding boxes


def draw_boxes(image, detection_results, resized_size, target_class="person"):
    original_size = image.size
    scale_w = original_size[0] / resized_size[0]
    scale_h = original_size[1] / resized_size[1]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    font_size = 10

    for i, bbox in enumerate(detection_results.boxes.xyxy):
        class_id = detection_results.boxes.cls[i]
        class_id = class_id.item() if isinstance(class_id, torch.Tensor) else class_id
        class_name = detection_results.names[class_id]

        if class_name == target_class:
            conf = detection_results.boxes.conf[i]  # Get the confidence score
            conf = conf.item() if isinstance(conf, torch.Tensor) else conf

            # Check if the confidence is greater than 0.5
            if conf > 0.5:
                # Scale the bounding box coordinates
                left, top, right, bottom = (
                    int(bbox[0].item() * scale_w),
                    int(bbox[1].item() * scale_h),
                    int(bbox[2].item() * scale_w),
                    int(bbox[3].item() * scale_h),
                )

                # Draw the bounding box
                draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)

                text = f"{class_name}: {conf:.2f}"
                text_width = draw.textlength(text, font=font)
                text_height = font_size

                text_position = (left, top - text_height)
                draw.rectangle([text_position, (left + text_width, top)], fill="red")
                draw.text((left, top - text_height), text, fill="white", font=font)

    return image


def resize_and_pad(image, stride=32, max_size=640):
    """
    Resize and pad the image to maintain aspect ratio and
    ensure dimensions are divisible by the YOLO model's stride.
    """
    # Resize the image, maintaining aspect ratio
    ratio = min(max_size / image.size[0], max_size / image.size[1])
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Pad the resized image to be divisible by the stride
    width, height = resized_image.size
    new_width = width if width % stride == 0 else width + stride - width % stride
    new_height = height if height % stride == 0 else height + stride - height % stride
    padded_image = ImageOps.expand(
        resized_image, border=(0, 0, new_width - width, new_height - height), fill=0
    )

    return padded_image


def apply_patch_to_image(
    image,
    patch,
    detection_results,
    resized_size,
    target_class="person",
    size_multiple=0.3,
):
    """
    Apply the adversarial patch to the image on the detected target class, maintaining the original aspect ratio of the patch
    and resizing it to cover a specified portion of the bounding box.

    :param image: PIL Image, the original image.
    :param patch: Adversarial patch as a PIL Image or a PyTorch tensor.
    :param detection_results: Results object from YOLO detection.
    :param resized_size: Size of the image used for detection (width, height).
    :param target_class: The class of objects to target with the patch.
    :param size_multiple: Fraction of the bounding box to cover with the patch.
    :return: Image with the adversarial patch applied.
    """
    original_size = image.size
    scale_w = original_size[0] / resized_size[0]
    scale_h = original_size[1] / resized_size[1]

    # Ensure the patch is a PIL Image
    if not isinstance(patch, Image.Image):
        pil_patch = to_pil_image(patch.squeeze().cpu())
    else:
        pil_patch = patch

    patched_image = image.copy()
    patch_aspect_ratio = pil_patch.width / pil_patch.height

    for i, bbox in enumerate(detection_results.boxes.xyxy):
        class_id = detection_results.boxes.cls[i]
        class_name = detection_results.names[
            class_id.item() if isinstance(class_id, torch.Tensor) else class_id
        ]
        if class_name == target_class:
            x1, y1, x2, y2 = (
                int(bbox[0].item() * scale_w),
                int(bbox[1].item() * scale_h),
                int(bbox[2].item() * scale_w),
                int(bbox[3].item() * scale_h),
            )

            box_width, box_height = x2 - x1, y2 - y1
            # Calculate the size of the patch to maintain aspect ratio and cover the specified portion of the box
            target_width = int(box_width * size_multiple)
            target_height = int(target_width / patch_aspect_ratio)

            if target_height > box_height * size_multiple:
                target_height = int(box_height * size_multiple)
                target_width = int(target_height * patch_aspect_ratio)

            # Resize patch to fit the calculated size
            resized_patch = pil_patch.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

            # Calculate the position to place the patch (centered within the bounding box)
            paste_x = x1 + (box_width - target_width) // 2
            paste_y = y1 + (box_height - target_height) // 2

            # Paste the resized patch onto the image
            patched_image.paste(resized_patch, (paste_x, paste_y))

    return patched_image
