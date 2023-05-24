from PIL import Image
def load_image_by_pil(file_name, respect_exif=False):
    file_name=file_name[0]
    if isinstance(file_name, str):
        image = Image.open(file_name).convert('RGB')
    elif isinstance(file_name, bytes):
        import io
        image = Image.open(io.BytesIO(file_name)).convert('RGB')
    else:
        print(f'jkjk',file_name)
    if respect_exif:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)
    return image

