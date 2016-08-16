for image in `ls data_papers/`
do
	echo "image ${image} : "
	image_res="${image%%.*}_res_hog_hsv_kmeans_all.jpg"
	echo "\begin{figure}[H]"
	echo "\begin{center}"
	echo "\includegraphics[scale=0.6]{../data_papers/$image_res}"
	echo "\end{center}"
	echo "\caption{de gauche à droite : image $image, binarisation, decomposition, vérité-terrain}"
	echo "\label{${image%%.*}}"
	echo "\end{figure}"
	echo "\clearpage"
	echo -e "\n"
done

