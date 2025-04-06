import React, { useState } from 'react';
import axios from 'axios';

const Combined = () => {
    // State variables for images and output
    const [image1, setImage1] = useState(null);
    const [image2, setImage2] = useState(null);
    const [isSubmitEnabled, setIsSubmitEnabled] = useState(false);
    const [outputImage, setOutputImage] = useState(null);
    const [categoryChanges, setCategoryChanges] = useState(null);

    // Handle image uploads
    const handleImageUpload = (event, setImage) => {
        const file = event.target.files[0];
        if (file) {
            setImage(URL.createObjectURL(file));
            checkIfSubmitEnabled(file, event.target.name);
        }
    };

    // Enable the submit button only when both images are uploaded
    const checkIfSubmitEnabled = (file, name) => {
        if (name === 'image1' && file) {
            setIsSubmitEnabled(image2 !== null);
        }
        if (name === 'image2' && file) {
            setIsSubmitEnabled(image1 !== null);
        }
    };

    const handleSubmit = async () => {
        if (image1 && image2) {
            const formData = new FormData();
            formData.append('file1', document.querySelector('input[name="image1"]').files[0]); // Use the actual file from the input
            formData.append('file2', document.querySelector('input[name="image2"]').files[0]); // Use the actual file from the input
    
            try {
                const response = await axios.post('http://localhost:8000/compare_images', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                });
    
                // Handle the response
                setOutputImage(response.data.combined_image);
                setCategoryChanges(response.data.category_changes);
            } catch (error) {
                console.error('Error uploading images:', error);
                alert('An error occurred while uploading the images.');
            }
        }
    };
    

    return (
        <div className="combined-page">
            <h1>Compare Images</h1>
            <div className="image-upload">
                <div>
                    <label htmlFor="image1">Upload Old Image:</label>
                    <input type="file" id="image1" name="image1" accept="image/*" onChange={(e) => handleImageUpload(e, setImage1)} />
                    {image1 && <img src={image1} alt="Old" style={{ width: '150px', marginTop: '10px' }} />}
                </div>
                <div>
                    <label htmlFor="image2">Upload New Image:</label>
                    <input type="file" id="image2" name="image2" accept="image/*" onChange={(e) => handleImageUpload(e, setImage2)} />
                    {image2 && <img src={image2} alt="New" style={{ width: '150px', marginTop: '10px' }} />}
                </div>
            </div>

            {/* Submit Button */}
            <button onClick={handleSubmit} disabled={!isSubmitEnabled} style={{ marginTop: '20px' }}>
                Submit
            </button>

            {/* Output Section */}
            {outputImage && (
                <div className="output-section" style={{ marginTop: '40px' }}>
                    <h3>Combined Image:</h3>
                    <img src={outputImage} alt="Combined" style={{ width: '400px', display: 'block', margin: '0 auto' }} />

                    <h3>Category Changes:</h3>
                    <ul>
                        {Object.entries(categoryChanges).map(([category, [percentage1, percentage2, change]], index) => (
                            <li key={index}>
                                <strong>{category}:</strong> {percentage1.toFixed(2)}%  {percentage2.toFixed(2)}% (Change: {change.toFixed(2)}%)
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default Combined;
