from argparse import ArgumentParser
import os
import tiny_face_eval
import video_Tiny_face
def test_image():
    argparse = ArgumentParser()
    argparse.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default="./mat2tf.pkl")
    argparse.add_argument('--data_dir', type=str, help='Image data directory.',
                          default="input_images")
    argparse.add_argument('--output_dir', type=str, help='Output directory for images with faces detected.',
                          default="output_images")
    argparse.add_argument('--prob_thresh', type=float, help='The threshold of detection confidence(default: 0.5).',
                          default=0.5)
    argparse.add_argument('--nms_thresh', type=float,
                          help='The overlap threshold of non maximum suppression(default: 0.1).', default=0.1)
    argparse.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=3)
    argparse.add_argument('--display', type=bool, help='Display each image on window.', default=False)

    args = argparse.parse_args()

    # check arguments
    assert os.path.exists(args.weight_file_path), "weight file: " + args.weight_file_path + " not found."
    assert os.path.exists(args.data_dir), "data directory: " + args.data_dir + " not found."
    assert os.path.exists(args.output_dir), "output directory: " + args.output_dir + " not found."
    assert args.line_width >= 0, "line_width should be >= 0."

    tiny_face_eval.evaluate(
        weight_file_path=args.weight_file_path, data_dir=args.data_dir, output_dir=args.output_dir,
        prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh,
        lw=args.line_width, display=args.display)


def test_videos():
    argparse = ArgumentParser()
    argparse.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default="./mat2tf.pkl")
    argparse.add_argument('--data_dir', type=str, help='Image data directory.', default="/home/asus/ai/yanhong.jia/video")
    argparse.add_argument('--output_dir', type=str, help='Output directory for images with faces detected.',
                          default="/home/asus/ai/yanhong.jia/facevideo")
    argparse.add_argument('--prob_thresh', type=float, help='The threshold of detection confidence(default: 0.5).',
                          default=0.5)
    argparse.add_argument('--nms_thresh', type=float,
                          help='The overlap threshold of non maximum suppression(default: 0.1).', default=0.1)
    argparse.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=3)
    argparse.add_argument('--display', type=bool, help='Display each image on window.', default=True)

    args = argparse.parse_args()

    # check arguments
    assert os.path.exists(args.weight_file_path), "weight file: " + args.weight_file_path + " not found."
    assert os.path.exists(args.data_dir), "data directory: " + args.data_dir + " not found."
    #assert os.path.exists(args.output_dir), "output directory: " + args.output_dir + " not found."
    assert args.line_width >= 0, "line_width should be >= 0."


    video_Tiny_face.evaluate(
        weight_file_path=args.weight_file_path, data_dir=args.data_dir, output_dir=args.output_dir,
        prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh,
        lw=args.line_width, display=args.display)



if __name__ == '__main__':
    #test_image()
    test_videos()