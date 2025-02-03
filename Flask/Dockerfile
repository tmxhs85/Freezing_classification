# Base image
FROM nginx:alpine

# Set working directory
WORKDIR /usr/share/nginx/html

# Copy frontend files
COPY index.html main.html ./

# Expose the port
EXPOSE 80

# Start NGINX
CMD ["nginx", "-g", "daemon off;"]
