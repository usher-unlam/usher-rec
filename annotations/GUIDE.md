## Configuración de entrenamiento en Nube

Las siguientes instrucciones definen bajo Ubuntu 18.10 la creacion del espacio de trabajo para crear los modelos de entrenamiento en Google Cloud

### Prerequisitos

Primero instalaremos Anaconda y aceptaremos todos los mensajes
```
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

Luego dependiendo del terminal que usemos ya sea bash o zsh
```
conda init <TERMINAL>
```

A continuacion crearemos la variable global, el espacio virtual de trabajo y lo activaremos.
```
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
conda create --name neuralfix_tf python=3
conda activate neuralfix_tf
```
Considerar que el espacio de trabajo de la guia es $HOME/Workspace



### Complementos
Una vez iniciado el entorno virtual con Anaconda (preferentemente Python3) instalaremos todos los complementos que se mencionan en la [guia de instalación] de Object Detection de Tensorflow
[Dropwizard](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) los cuales son

```
pip install tensorflow protobuf Cython contextlib2 pillow jupyter matplotlib filelock
```
Tambien agrego vcode como una herramienta sencilla para desarrollar o retocar los respoitorios que descargaremos despues
```
wget -q https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb && sudo apt-get install apt-transport-https && sudo apt-get update && sudo apt-get install dotnet-sdk-2.1
```

### Repositorios
Crearemos la carpeta de espacio de trabajo y luego tendremos que clonar y modificar algunos repositorios los cuales son:

```
mkdir $HOME/Workspace && cd $HOME/Workspace && mkdir google && cd google
sudo apt-get update && sudo apt-get install git-core && git --version
git clone https://github.com/tensorflow/models.git
cd .. && git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI && make && cp -r pycocotools $HOME/Workspace/google/models/research
```
Ademas crearemos los vinculos de protobuf en el repositorio clonado
```
cd $HOME/Workspace/google/models/research/ && wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
```
y por ultimo crearemos la variable de entorno
```
cd $HOME/Workspace/google/models/research/ 
echo 'export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim' >> ~/.bashrc
```

Verificamos si todo salio bien
```
python object_detection/builders/model_builder_test.py
```

sudo apt-get -y install build-essential checkinstall libx11-dev libxext-dev zlib1g-dev libpng-dev libjpeg-dev libfreetype6-dev libxml2-dev

### Otras herramientas
Imagemagick es una potente herramienta para editar imagenes. Si la desean instalar pueden correr el siguiente script en la terminal
```
sudo apt remove -y imagemagick
sudo apt-get -y build-dep imagemagick
mkdir $HOME/Workspace/imagemagick_build && cd $HOME/Workspace/imagemagick_build
wget http://www.imagemagick.org/download/ImageMagick.tar.gz && tar xvf ImageMagick.tar.gz && IMAGEMAGICKFOLDER=$(ls | grep ImageMagick-) && IMAGEMAGICKVER=$(echo $IMAGEMAGICKFOLDER | cut -c13-20) cd $IMAGEMAGICKFOLDER && ./configure \
--with-png=yes \
--with-jpeg=yes \
--with-jp2=yes \
--with-tiff=yes \
--with-freetype=yes && make clean && make && sudo checkinstall -D --install=yes \
--fstrans=no \
--pkgname imagemagick \
--backup=no \
--deldoc=yes \
--deldesc=yes \
--delspec=yes \
--default \
--pkgversion "$IMAGEMAGICKVER" > ../log.txt 2> ../log.txt
make distclean && sudo ldconfig /usr/local/lib
rm -rf $HOME/Workspace/imagemagick_build
```

LabelImg es un excelente anotador de Bounding Boxes el cual se instala asi:
```
sudo apt-get install pyqt4-dev-tools && pip install lxml labelImg
sudo apt-get install python3-pyqt5
git clone https://github.com/tzutalin/labelImg.git
cd labelImg-master
make qt5py3
```

## Entrenamiento Usher Remoto
El entrenamiento remoto consiste en seguir los pasos descriptos por google [aqui](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md) :


### Para crear el record de entrenamiento
```
python models/research/object_detection/dataset_tools/create_pascal_tf_record.py \
   	--data_dir=$HOME/Workspace/neuralfix_usher/package_cdbsas_01/train \
   	--year=VOC2012 \
   	--output_path=$HOME/Workspace/neuralfix_usher/package_cdbsas_01/train/training.record  \
   	--label_map_path=$HOME/Workspace/neuralfix_usher/package_cdbsas_01/labels.pbtxt \
   	--set=train
```

### Para crear el record de evaluacion

```
python models/research/object_detection/dataset_tools/create_pascal_tf_record.py \
   	--data_dir=$HOME/Workspace/neuralfix_usher/package_cdbsas_01/eval \
   	--year=VOC2012 \
   	--output_path=$HOME/Workspace/neuralfix_usher/package_cdbsas_01/eval/eval.record  \
   	--label_map_path=$HOME/Workspace/neuralfix_usher/package_cdbsas_01/labels.pbtxt \
   	--set=val
```



### Configurar Object Detection para entrenar en a nube
Luego tenemos que generar los paquetes que se subiran a Google Cloud con Object Detection configurado

```
cd $HOME/Workspace/google/models/research/ 
bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
python setup.py sdist
(cd slim && python setup.py sdist)
cp -r /tmp/pycocotools/pycocotools-2.0.tar.gz $HOME/Workspace/google/models/research/dist
```
Necesitaremos el paquete de Google SDK para poder acceder desde la API a los servicios de Google Cloud.
Opcionalmente se decidio usar el SDK en Python. Ustedes tienen libertad en cual usar
```
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
google-cloud-sdk-app-engine-python
```

Necesitaremos iniciar sesion. Esto abrira en el navegador nuestra cuenta de Google para habilitar permisos
```
gcloud auth application-default login
gcloud auth login
```

Por ultimo configuramos el proyecto de Google Cloud que hayamos creado al iniciar la cuenta en el navegador y nos aseguraremos que este tdo configurado
```
gcloud config set project xxxxx-xxxxxxxxx-0000000
gcloud config list
```

## Ejecucion de entrenamiento

De esta forma una vez que tengamos todos los archivos que contemplan los preparativos ejecutaremos el entrenamiento con este comando
```
cd $HOME/Workspace/google/models/research/ 
gcloud ml-engine jobs submit training neuralfix_usher_cd_bsas_01_01_`date +%m_%d_%Y_%H_%M_%S` \
   --runtime-version 1.12 \
   --job-dir=gs://neuralfix_usher_cd_bsas_01/train \
   --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,dist/pycocotools-2.0.tar.gz \
   --module-name object_detection.model_main \
   --region us-west1 \
   --config object_detection/samples/cloud/cloud.yml \
   -- \
   --pipeline_config_path=gs://neuralfix_usher_cd_bsas_01/data/pipeline.config \
   --eval_dir=gs://neuralfix_usher_cd_bsas_01/train/eval \
   --model_dir=gs://neuralfix_usher_cd_bsas_01/train
```
Es importante recordar que cloud.yml tiene la version de modelos de tensorflow y debe coninsidir con la version de runtime que ejecutaremos


Para visualizar los resultados podemos ver la consola de Log o Tensorboard
```
https://console.cloud.google.com/ai-platform/jobs?project=xxxxx-xxxxxxxxx-0000000
tensorboard --logdir=gs://neuralfix_usher_cd_bsas_01
```

Descargar modelos entrenados

Una vez realizado el entrenamiento con est script podremos descargar los modelos segun el STEP deseado y obtener el modelo de inferencia:
```
BUCKET=gs://neuralfix_usher_cd_bsas_01
STEP=85961
FOLDER_NAME=$(echo $BUCKET | cut -f3 -d"/")_`date +%Y_%m_%d`_$STEP
echo $FOLDER_NAME
cd $HOME/Workspace/google/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
gsutil cp $BUCKET/train/model.ckpt-$STEP.data-00000-of-00003 .
gsutil cp $BUCKET/train/model.ckpt-$STEP.data-00001-of-00003 .
gsutil cp $BUCKET/train/model.ckpt-$STEP.data-00002-of-00003 .
gsutil cp $BUCKET/train/model.ckpt-$STEP.index .
gsutil cp $BUCKET/train/model.ckpt-$STEP.meta .
gsutil cp $BUCKET/data/pipeline.config .
gsutil cp $BUCKET/data/labels.pbtxt .
mkdir $FOLDER_NAME
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path pipeline.config \
    --trained_checkpoint_prefix model.ckpt-$STEP \
    --output_directory ./$FOLDER_NAME
rm model.ckpt-*
```