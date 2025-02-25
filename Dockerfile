# Use the official Python image as a base
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy the application files
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Waitress will serve on
EXPOSE 8080

# Set environment variables
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_BUCKET_NAME=${AWS_BUCKET_NAME}
ENV AWS_REGION=${AWS_REGION}
ENV PINECONE_API_KEY=${PINECONE_API_KEY}
ENV PINECONE_ENV=${PINECONE_ENV}
ENV PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}    

# Run the application with Waitress (Production WSGI Server)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "app:app"]
