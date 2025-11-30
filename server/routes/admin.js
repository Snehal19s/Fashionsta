const express = require('express');
const multer = require('multer');
const { storage } = require('../utils/cloudinary');
const Product = require('../models/Product'); // your product schema

const upload = multer({ storage });

const router = express.Router();

// Add Product
router.post('/admin/products', upload.array('images'), async (req, res) => {
    try {
        const imageUrls = req.files.map(f => f.path); // Cloudinary image URLs
        const sizes = req.body.sizes?.split(',').map(s => s.trim()) || [];
        const bodyShapeTags = req.body.bodyShapeTags?.split(',').map(t => t.trim()) || [];
        const skinToneTags = req.body.skinToneTags?.split(',').map(t => t.trim()) || [];

        const newProduct = new Product({
            name: req.body.name,
            description: req.body.description,
            price: parseFloat(req.body.price),
            category: req.body.category,
            gender: req.body.gender,
            stock: parseInt(req.body.stock),
            sizes,
            bodyShapeTags,
            skinToneTags,
            images: imageUrls,
        });

        await newProduct.save();
        res.status(201).json({ message: 'Product added', product: newProduct });
    } catch (err) {
        res.status(500).json({ message: 'Error adding product', error: err.message });
    }
});
