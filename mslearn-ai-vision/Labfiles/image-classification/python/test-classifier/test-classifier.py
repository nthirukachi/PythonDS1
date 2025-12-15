# """
# This Python script tests an image classification model trained with Azure Custom Vision.
# It loads test images, sends them to the prediction API, and prints out the predicted labels with high confidence.
# You need to set up your Azure Custom Vision project and provide API credentials in a .env file.
# """
# Import the CustomVisionPredictionClient to interact with Azure Custom Vision Prediction API
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
# Import authentication class to use API keys for Azure services
from msrest.authentication import ApiKeyCredentials
# Import os module to interact with the operating system (e.g., file paths, environment variables)
import os

# Main function: Loads configuration, authenticates with Azure, and classifies test images using the trained model.
def main():
    # Import load_dotenv to load environment variables from a .env file
    from dotenv import load_dotenv

    # Clear the console for better readability (Windows uses 'cls', others use 'clear')
    os.system('cls' if os.name=='nt' else 'clear')

    try:  # Try block to catch and print any errors
        # Load environment variables from .env file
        load_dotenv()
        # Get the Azure Custom Vision prediction endpoint from environment
        prediction_endpoint = os.getenv('PredictionEndpoint')
        # Get the prediction key (API key) from environment
        prediction_key = os.getenv('PredictionKey')
        # Get the project ID for your Custom Vision project
        project_id = os.getenv('ProjectID')
        # Get the model name to use for predictions
        model_name = os.getenv('ModelName')

        # Authenticate a client for the prediction API using your API key
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

        # Classify each image in the 'test-images' folder
        for image in os.listdir('test-images'):
            # Read the image file as binary data
            image_data = open(os.path.join('test-images',image), "rb").read()
            # Send the image to the prediction API and get results
            results = prediction_client.classify_image(project_id, model_name, image_data)

            # Loop over each label prediction and print any with probability > 50%
            for prediction in results.predictions:
                # If the model is more than 50% confident about a label, print it
                if prediction.probability > 0.5:
                    print(image, ': {} ({:.0%})'.format(prediction.tag_name, prediction.probability))
    except Exception as ex:  # If any error occurs, print it
        print(ex)

# If this script is run directly (not imported), call the main function to start the workflow
if __name__ == "__main__":
    main()

