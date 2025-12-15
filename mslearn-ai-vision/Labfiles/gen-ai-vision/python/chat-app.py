# """
# This Python script is a simple AI-powered chat application for a grocery store.
# It uses Azure AI and OpenAI to answer questions about produce, including image-based queries.
# You need to set up Azure credentials and provide API settings in a .env file.
# """
# Import the os module to interact with the operating system (e.g., clearing the console, environment variables)
import os
# Import urlopen and Request to fetch images from the web
from urllib.request import urlopen, Request
# Import base64 to encode image data for sending to the AI model
import base64
# Import Path for file path operations (not used in this script, but often useful)
from pathlib import Path
# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Import Azure identity and AI project client for authentication and project management
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
# Import AzureOpenAI for interacting with OpenAI models on Azure
from openai import AzureOpenAI

# Main function: Loads configuration, authenticates with Azure, and runs the chat loop for user questions.
def main(): 

    # Clear the console for better readability (Windows uses 'cls', others use 'clear')
    os.system('cls' if os.name=='nt' else 'clear')
        
    try:  # Try block to catch and print any errors
    
        # Load environment variables from .env file
        load_dotenv()
        # Get the Azure AI project endpoint from environment
        project_endpoint = os.getenv("PROJECT_CONNECTION")
        # Get the model deployment name from environment
        model_deployment =  os.getenv("MODEL_DEPLOYMENT")


        # Initialize the Azure AI project client for accessing AI services
        project_client = AIProjectClient(            
                credential=DefaultAzureCredential(
                    exclude_environment_credential=True,
                    exclude_managed_identity_credential=True
                ),
                endpoint=project_endpoint,
            )
        

        # Get a chat client for interacting with the OpenAI model
        openai_client = project_client.get_openai_client(api_version="2024-10-21")



        # Set up the system message to instruct the AI about its role
        system_message = "You are an AI assistant in a grocery store that sells fruit. You provide detailed answers to questions about produce."
        prompt = ""  # Initialize the prompt variable

        # Start a loop to interact with the user until they type 'quit'
        while True:
            # Ask the user for a question about the image
            prompt = input("\nAsk a question about the image\n(or type 'quit' to exit)\n")
            if prompt.lower() == "quit":  # Exit the loop if the user types 'quit'
                break
            elif len(prompt) == 0:
                print("Please enter a question.\n")  # Prompt user to enter a question if input is empty
            else:
                print("Getting a response ...\n")  # Inform the user that a response is being generated

                # Set the image URL to send to the AI model
                image_url = "https://github.com/MicrosoftLearning/mslearn-ai-vision/raw/refs/heads/main/Labfiles/gen-ai-vision/orange.jpeg"
                image_format = "jpeg"  # Specify the image format
                # Create a request object to fetch the image from the web
                request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
                # Read the image data and encode it in base64 for the AI model
                image_data = base64.b64encode(urlopen(request).read()).decode("utf-8")
                # Create a data URL for the image to send to the AI model
                data_url = f"data:image/{image_format};base64,{image_data}"

                # Send the user's question and the image to the AI model and get a response
                response = openai_client.chat.completions.create(
                    model=model_deployment,
                    messages=[
                        {"role": "system", "content": system_message},
                        { "role": "user", "content": [  
                            { "type": "text", "text": prompt},
                            { "type": "image_url", "image_url": {"url": data_url}}
                        ] } 
                    ]
                )
                # Print the AI's response to the user
                print(response.choices[0].message.content)                    


    except Exception as ex:  # If any error occurs, print it
        print(ex)


# If this script is run directly (not imported), call the main function to start the workflow
if __name__ == '__main__': 
    main()