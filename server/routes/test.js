const express = require('express');
const { v2: cloudinary } = require('cloudinary');
const router = express.Router();

router.get('/test-cloudinary', (req, res) => {
    res.json({
        cloud_name: cloudinary.config().cloud_name,
        api_key: cloudinary.config().api_key,
    });
});

module.exports = router;
