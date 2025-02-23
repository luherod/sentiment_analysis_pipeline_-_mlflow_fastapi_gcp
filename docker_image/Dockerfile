FROM python:3.11.2-slim  
  
# Set the working directory in the container  
WORKDIR /app  
  
# Install pip requirements  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt 

# Download spaCy model
RUN python -m spacy download en_core_web_sm
  
# Copy the content of the local src directory to the working directory  
COPY . .  
  
# Specify the port number the container should expose  
EXPOSE 8080  
  
# Run the application  
CMD ["fastapi", "run", "main.py", "--port", "8080"]