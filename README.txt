
A joint study of hand tracking and hand writing recognition

Files and their use:

	1) code_htr_curve/extract_points : extracts points from svg files and gives out a .npy file.

	2) plot_points : given a point cloud plots the point in 2D space.
	
	3) show_unet_image : given a mask converts it into image and displays it.
	
	4) point_cloud_label : labels a point cloud.
	
	5) point_image : converts an image into segmentation binary mask with foreground black and background white
	
	6) prepare_dataset : used to prepare hand videos dataset
	
	7) hand : to render hand keypoints on the hand frames.
	
	8) code_htr_curve/model : contains the model architectures, final models mdl7 and mdl16

	9) code_htr_curve/train_main_1 : train file for training model to predict using only pre-computed features from pointnet and hwnet
	
	10) code_htr_curve/train_main_2 : train file for training model to predict using partial pointnet and pre-computed features from hwnet
	
	11) code_htr_curve/hand_track : train file for training model to predict hand keypoints
	
	12) code_htr_curve/transformer2 : contains the transformer model used in hand_track
	
	13) code_htr_curve/word_stroke : train file for training binary unet
	
	14) code_htr_curve/unet : conatins the unet model for binary segmentation
	
	15) code_htr_curve/train_labeled_unet : train file for training final unet model using the latent space from binary unet

	16) code_htr_curve/dataset : conatins function to read video frames and masks for training unet

	17) code_htr_curve/utils : contains some useful functions
	
Different Models used in the project:

1) Virtual sketching: 

	a) Use: It is used to convert images to vector images.
 
	b) How it does: https://github.com/MarkMoHR/virtual_sketching
		
	c) Useful files:
		i) test_vectorization.py : runs the model on the given image folder  given in the form of ann file.
		ii) svg_conversion.py : converts the created npz files to svg files.
		
2) HwNet:

	a) Use: gives out feature representation for images given in ann file.
	
	b) How it does so: refer to https://github.com/kris314/hwnet
	
3) PointNet:

	a) Use: gives out feture representationof point cloud
	
	b) What it does : refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch
	
	c) Useful files: 
		i) test_classification.py : runs the pretrained model to get feature vector for the point cloud npy file.
		
	
Code Flow: 

1) image recognition/ word spotting

	a) dataset creation:
		images -> virtual_sketching/test_vectorization -> npz -> virtual_sketching/svg_conversion -> svgs -> code_htr_curve/extract_points ->
		point cloud -> point_cloud_label -> labeled point cloud
	
	b) code run:
		labeled_point_cloud -> train file -> output
	
2) text+image to keypoint

	a) dataset creation
	
		prepare_dataset -> videos + keypoints + indices