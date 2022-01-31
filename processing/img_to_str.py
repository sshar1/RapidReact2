import base64

# Converts image to string using base64 encoding.
def to_b64(filename):
    with open(filename, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())

    return my_string