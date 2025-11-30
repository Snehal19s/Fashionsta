// require('dotenv').config();
// const express = require('express');
// const mongoose = require('mongoose');
// const multer = require('multer');
// const { v2: cloudinary } = require('cloudinary');
// const { CloudinaryStorage } = require('multer-storage-cloudinary');

// const app = express();
// app.use(express.static('public'));
// app.use(express.json());

// // MongoDB connection
// mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
//   .then(() => console.log('Connected to MongoDB Atlas'))
//   .catch(err => console.error('MongoDB connection error:', err));

// // Schema
// const Image = mongoose.model('Image', new mongoose.Schema({
//   url: String,
//   name: String
// }));

// // Cloudinary config
// cloudinary.config({
//   cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
//   api_key: process.env.CLOUDINARY_API_KEY,
//   api_secret: process.env.CLOUDINARY_API_SECRET
// });

// // Multer + Cloudinary Storage
// const storage = new CloudinaryStorage({
//   cloudinary: cloudinary,
//   params: {
//     folder: 'my_images',
//     allowed_formats: ['jpg', 'png']
//   }
// });

// const upload = multer({ storage });

// // Upload route
// app.post('/upload', upload.single('image'), async (req, res) => {
//   const image = new Image({
//     url: req.file.path,
//     name: req.file.originalname
//   });
//   await image.save();
//   res.json({ message: 'Image uploaded successfully!', image });
// });

// // Serve static files from uploads folder
// app.use('/uploads', express.static('uploads'));

// // Start server
// app.listen(3000, () => {
//   console.log('Server running on http://localhost:3000');
// });

app.use('/api/test', require('./routes/test'));

---