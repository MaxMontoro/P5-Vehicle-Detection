from utils import *
from sliding_window import *

test_out_file = '_full_project_video_out.mp4'
clip_test = VideoFileClip('project_video.mp4')
clip_test_out = clip_test.fl_image(process_image)
clip_test_out.write_videofile(test_out_file, audio=False)
