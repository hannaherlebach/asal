from .llava import LLAVA

def create_video_text_model(vtm_name):
    """
    Create the video text model given a video text model name.
    """
    if vtm_name=='llava':
        vtm = LLAVA()
    else:
        raise ValueError(f"Unknown foundation model name: {vtm_name}")
    return vtm