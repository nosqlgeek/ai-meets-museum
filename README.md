
# ![Logo](https://github.com/nosqlgeek/ai-meets-museum/assets/63855548/6039b0dd-1a78-464b-92c6-746099e0e966)

This project is a joint effort of the [University of Applied Sciences Augsburg](https://www.hs-augsburg.de), [NoSQL Geeks](https://www.nosqlgeeks.com/de/index.html), and the [local museum in Krumbach (Schwaben)](https://www.museum-krumbach.de).

*  **Subject**: AI-powered tool for simplified information gathering around artifacts in a museum

*  **Description**: Museums of any size are challenged to gather details about artifacts by having a limited number of human resources available. Details such as the class of an artifact, the material, the color, the age, the provenance are part of such an artifact description. The goal of this project is to create an Open Source Software tool that automatically detects properties, such as the shape, the color or the material.

*  **Implementation idea**: We will evaluate several Machine Learning models in order to create vector embeddings. Finally Redis' Vector Similarity Search feature will be used to identify similarities between already classified and new artifacts.

## Getting Started
- <a href="https://github.com/nosqlgeek/ai-meets-museum/blob/main/src/aimm/docker-setup.md" target="_blank">Installation Guide</a>
- [Installation Guide](https://github.com/nosqlgeek/ai-meets-museum/blob/main/src/aimm/docker-setup.md)
- [Product Overview](https://showcase.informatik.hs-augsburg.de/sose-2023/ai-meets-museum)

## Technology Stack

![technologien](https://github.com/nosqlgeek/ai-meets-museum/assets/63855548/b964f6e1-929a-4b48-9847-6ccc4370587a)


### Backend

#### ResNet50
We utilized the powerful ResNet50 model to generate image vector data from our source images. Leveraging the state-of-the-art deep learning capabilities of ResNet50, we have extracted rich feature representations, allowing us to efficiently store and manage our image data in our Redis database. By combining the strengths of ResNet50 and Redis, we have created a robust solution for handling image data, enabling seamless retrieval and integration within our projects.

#### Redis
With Redis, we have unlocked a robust and scalable solution for handling large volumes of data, ensuring optimal performance and seamless integration within our projects. Through our repository, you can explore our codebase, discover innovative implementations, and gain insights into how we harness Redis's capabilities to enhance data storage and retrieval.

#### Vector Embedding
Vector embedding refers to the process of representing objects, such as images, text, or other data, as fixed-length numerical vectors in a continuous vector space. In the context of Redis, vector embedding enables us to store and manipulate object data efficiently.

Redis, being an in-memory data structure store, is primarily designed for key-value storage. However, by leveraging vector embedding techniques, we can represent complex object data as compact numerical vectors that can be stored and retrieved from Redis with ease.

By using Redis to store vector embeddings, we gain several advantages. First, the compact nature of vector representations allows for efficient storage and retrieval, as the numerical vectors occupy less space compared to the original object data. Second, Redis provides fast and scalable operations, enabling quick access to the vector embeddings. This facilitates various operations, such as similarity search, recommendation systems, and data analytics, which can be performed efficiently using the stored vector data in Redis.

### Frontend

#### Flask
e have leveraged the power of Flask to build an intuitive and user-friendly frontend implementation. Using Flask as the foundation, we have created a seamless bridge between our frontend and backend services, including the robust Redis database. With Flask, we have crafted a dynamic and responsive user interface that effortlessly connects users to the powerful functionality of our backend services.

#### Bootstrap
We employed the versatile Bootstrap framework to create a visually appealing and responsive frontend design. With Bootstrap, we have harnessed a comprehensive set of CSS and JavaScript components, allowing us to effortlessly build a modern and consistent user interface. By leveraging Bootstrap's grid system, responsive utilities, and extensive library of pre-built components, we have ensured a seamless user experience across various devices and screen sizes.

## Features
- suchfunktion

## Prerequisites 

Preparation material will be made available here: 

* [Preparation](./doc/Preparation.md)  

Please join the following Discord server for Q&A:  

* [Discord](https://discord.gg/J2qERxHCPP)  

You will first only be able to communicate via the `#lobby` channel. We will then add you to the private channel `#ai-meets-museum` for any project-specific communication.  

You will also need the source datab and a Redis Stack database. You can either use a local Redis Stack database, or one of the Cloud databases that were prepared for this project. The database endpoints can be found here:  

* [Databases](https://github.com/nosqlgeek/ai-meets-museum-priv)  

Further details about the source images can be found here:  

* [Source images](./doc/SourceImages.md)  

## Functional Requirements  

A list of the functional requirements can be found here: 

* [Requirements](./doc/Requirements.md)  

## Tasks and Project Management  

We will use Github tasks and a Kanban-like board to keep an overview of the work that needs to be done. You can find the associated Github project here:  

* [Project](https://github.com/users/nosqlgeek/projects/1)
