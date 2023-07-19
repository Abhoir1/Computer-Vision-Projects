import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
ins.segmentImage("assets/image1.jpeg", show_bboxes=False, output_image_name="results/already_trained1.jpg")
ins.segmentImage("assets/image2.jpeg", show_bboxes=False, output_image_name="results/already_trained2.jpg")
ins.segmentImage("assets/image3.jpeg", show_bboxes=False, output_image_name="results/already_trained3.jpg")
ins.segmentImage("assets/image4.jpeg", show_bboxes=False, output_image_name="results/already_trained4.jpg")
ins.segmentImage("assets/image5.jpg", show_bboxes=False, output_image_name="results/already_trained5.jpg")
