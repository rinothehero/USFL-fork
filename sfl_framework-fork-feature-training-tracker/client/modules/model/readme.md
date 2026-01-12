# client/modules/model

Since Pickle does not serialize class definitions, customized model information is stored inside the `modules/model` directory. For non-customized models (i.e., models imported by the torchvision or transform library), the model structure is not stored separately as it is already included within the library.

Note: This folder should be located in the "same position" relative to both the server and the client. For example, if the server's `modules/model` is located at `server/modules/model`, then the client's should be located at `client/modules/model`.
