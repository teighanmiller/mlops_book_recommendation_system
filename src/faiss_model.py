"""
Contains the FaissModel for logging in mlflow
"""
import tempfile
import mlflow
import boto3
import faiss
import numpy as np


class FaissModel(mlflow.pyfunc.PythonModel):
    """
    Faiss pyfunc model
    """

    def __init__(self):
        self.index = None

    def load_context(self, context):
        """
        Loading context
        """
        index_path = context.artifacts["faiss_index"]
        bucket, key = self._parse_s3_uri(index_path)

        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile() as f:
            s3.download_file(bucket, key, f.name)
            self.index = faiss.read_index(f.name)

    def predict(self, context, model_input):
        """
        Getting predictions
        """
        if isinstance(model_input, np.ndarray):
            data = model_input
        else:
            data = model_input.to_numpy()
        _, labels = self.index.search(data.astype(np.float32), 1)
        print(context)
        return labels

    def _parse_s3_uri(self, s3_uri):
        """
        parser for s3 uri
        """
        assert s3_uri.startswith("s3://")
        parts = s3_uri[5:].split("/", 1)
        return parts[0], parts[1]
