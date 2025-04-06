import React, { useState } from 'react';
import axios from 'axios';

const Land = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [overlayedImage, setOverlayedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
    setOverlayedImage(null);
  };

  const handleUpload = async () => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:8000/overlay', formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const imageUrl = URL.createObjectURL(response.data);
      setOverlayedImage(imageUrl);
    } catch (error) {
      alert('Error uploading image: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-2xl mx-auto text-center">
      <h1 className="text-2xl font-bold mb-4">Land Use Map Generator</h1>

      <input type="file" accept="image/png, image/jpeg" onChange={handleImageChange} />
      <button
        onClick={handleUpload}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
        disabled={!selectedImage || loading}
      >
        {loading ? 'Processing...' : 'Upload and Generate The Map'}
      </button>

      {selectedImage && (
        <div className="mt-6">
          <h2 className="font-semibold mb-2">Original Image:</h2>
          <img
            src={URL.createObjectURL(selectedImage)}
            alt="Selected"
            className="max-w-full max-h-64 mx-auto"
          />
        </div>
      )}

      {overlayedImage && (
        <div className="mt-6">
          <h2 className="font-semibold mb-2">Land use Map:</h2>
          <img src={overlayedImage} alt="Overlayed" className="max-w-full max-h-96 mx-auto" />
        </div>
      )}
    </div>
  );
};

export default Land;
