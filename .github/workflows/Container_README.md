# Container Use Instructions
## Podman Desktop App *
 
 When using the Podman desktop app to create the container image :
1. Open the images tab and select the build option

![alt text](container_instruct_images/image_creation.png)

2. In the Containterfile path browse to the location of the Dockerfile in .gihub/workflows and select the file

![alt text](container_instruct_images/Build-screen.png)

3. Enter a name for the image and click the build button
4. Once the image is built it will appear in the images tab. To create a container from the image click the run button

![alt text](container_instruct_images/image-created-tab.png)

5. Give the container a name and mount the pymgm repo using the Volume option to the location of the code on the host machine.
   For the path inside the container enter `/app` 

![alt text](container_instruct_images/create-container-and-mount.png)

6. Scroll down and press start container to start the container. \
<sub><sub>*This assumes that Podman desktop app is installed and a Podman machine is created and configured</sub><sub>
## Podman CLI

1. Build the container image from the Dockerfile using the command:
````
podman build -f /path/to/Dockerfile -t image_name .
````
where /path/to/Dockerfile is the path to the Dockerfile located in .github/workflows and image_name is the desired name for the container image.



2. Once the image is built, you can list it using:

podman images

3. To create a container from the image, run:


````
podman run -it --name container_name -v /path/to/pymgm:/app image_name
````

3. Replace container_name with the desired name for your container, image_name with the name of the image you created earlier,
and /path/to/pymgm with the path to your pymgm repo on the host machine.*

<sub>*If running using a WSL or gitbash on a Windows machine be sure to add an extra ' \ ' for every ' \ ' in the path.  
EX) C:\User\username should be replaced with C:\\User\\username<sub>



   


4.Start the container using the command:
````
podman start container_name
````
This will start the container using the previously created name container_name.
