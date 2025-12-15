# """
# This Python script automates the process of training an image classification model using Azure Custom Vision.
# It uploads tagged images from folders, trains the model, and provides status updates.
# You need to set up your Azure Custom Vision project and provide API credentials in a .env file.
# """
# Import the CustomVisionTrainingClient to interact with Azure Custom Vision Training API
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
# Import models needed for uploading images and tagging regions
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
# Import authentication class to use API keys for Azure services
from msrest.authentication import ApiKeyCredentials
# Import time module to add delays while waiting for training to complete
import time
# Import os module to interact with the operating system (e.g., file paths, environment variables)

## Main function: Loads configuration, authenticates with Azure, uploads images, and trains the model.
import os

def main():
    # Import load_dotenv to load environment variables from a .env file
    from dotenv import load_dotenv
    # Declare global variables so they can be used in other functions
    global training_client
    global custom_vision_project

    # Clear the console for better readability (Windows uses 'cls', others use 'clear')
    os.system('cls' if os.name=='nt' else 'clear')

    try:  # Try block to catch and print any errors
        # Load environment variables from .env file
        load_dotenv()
        # Get the Azure Custom Vision training endpoint from environment
        training_endpoint = os.getenv('TrainingEndpoint')
        # Get the training key (API key) from environment
        training_key = os.getenv('TrainingKey')
        # Get the project ID for your Custom Vision project
        project_id = os.getenv('ProjectID')

        # Authenticate a client for the training API using your API key
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        training_client = CustomVisionTrainingClient(training_endpoint, credentials)

        # Get the Custom Vision project object using the project ID
        custom_vision_project = training_client.get_project(project_id)

        # Upload and tag images from the specified folder
        Upload_Images('more-training-images')

        # Train the model with the uploaded images
        Train_Model()
        
    except Exception as ex:  # If any error occurs, print it
        print(ex)

## Upload_Images: Uploads images from the specified folder to Azure Custom Vision and tags them according to folder names.
def Upload_Images(folder):
    print("Uploading images...")  # Inform the user that images are being uploaded
    tags = training_client.get_tags(custom_vision_project.id)  # Get all tags defined in the project
    for tag in tags:  # Loop through each tag
        print(tag.name)  # Print the tag name
        # For each tag, loop through images in the corresponding folder
        for image in os.listdir(os.path.join(folder,tag.name)):
            # Read the image file as binary data
            image_data = open(os.path.join(folder,tag.name,image), "rb").read()
            # Upload the image to the project and associate it with the current tag
            training_client.create_images_from_data(custom_vision_project.id, image_data, [tag.id])

## Train_Model: Starts the training process and waits until the model is fully trained, printing status updates.
def Train_Model():
    print("Training ...")  # Inform the user that training has started
    iteration = training_client.train_project(custom_vision_project.id)  # Start training the model
    # Wait until training is complete
    while (iteration.status != "Completed"):
        # Get the latest status of the training iteration
        iteration = training_client.get_iteration(custom_vision_project.id, iteration.id)
        print (iteration.status, '...')  # Print the current status
        time.sleep(5)  # Wait for 5 seconds before checking again
    print ("Model trained!")  # Inform the user that training is finished


if __name__ == "__main__":  # If this script is run directly (not imported)
    main()  # Call the main function to start the workflow


