const express = require('express');
const cors = require('cors');
const path = require('path');
const config = require('./config');
const connectDB = require('./config/db');

// Import routes
const productRoutes = require('./routes/productRoutes');
const adminRoutes = require('./routes/adminRoutes');
const authRoutes = require('./routes/authRoutes');
const userRoutes = require('./routes/userRoutes');
const orderRoutes = require('./routes/orderRoutes');
const reviewRoutes = require('./routes/reviewRoutes');

// Initialize Express app
const app = express();
require('dotenv').config();

// Connect to Database
connectDB();

// Middleware to parse JSON and urlencoded data
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// API Routes
app.use('/api/products', productRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/orders', orderRoutes);
app.use('/api/reviews', reviewRoutes);

// Basic error handler
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send({ message: err.message || 'Something broke!' });
});
app.use('/api/test', require('./routes/test'));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
// Start server
const PORT = config.PORT || 5001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
app.use('/api/products', require('./routes/recommendationRoute'));