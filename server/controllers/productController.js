const Product = require('../models/Product');
const User = require('../models/User'); // For recommendations

// @desc    Fetch all products or by category/gender/search/filter
// @route   GET /api/products
// @access  Public
exports.getProducts = async (req, res) => {
    const pageSize = 55000; // Products per page
    const page = Number(req.query.pageNumber) || 1;

    const keyword = req.query.keyword ? {
        name: { $regex: req.query.keyword, $options: 'i' }
    } : {};

    const categoryFilter = req.query.category ? { category: req.query.category } : {};
    const genderFilter = req.query.gender ? { gender: req.query.gender } : {};
    // Add more filters for price range, size, rating
    const priceFilter = {};
    if (req.query.minPrice) priceFilter.price = { ...priceFilter.price, $gte: Number(req.query.minPrice) };
    if (req.query.maxPrice) priceFilter.price = { ...priceFilter.price, $lte: Number(req.query.maxPrice) };

    const ratingFilter = req.query.rating ? { ratings: { $gte: Number(req.query.rating) } } : {};
    // Size filter needs to check if req.query.size is in product.sizes array
    const sizeFilter = req.query.size ? { sizes: req.query.size } : {};


    try {
        const count = await Product.countDocuments({ ...keyword, ...categoryFilter, ...genderFilter, ...priceFilter, ...ratingFilter, ...sizeFilter });
        const products = await Product.find({ ...keyword, ...categoryFilter, ...genderFilter, ...priceFilter, ...ratingFilter, ...sizeFilter })
            .limit(pageSize)
            .skip(pageSize * (page - 1))
            .sort({ createdAt: -1 }); // Sort by newest

        res.json({ products, page, pages: Math.ceil(count / pageSize) });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

// @desc    Fetch single product by ID
// @route   GET /api/products/:id
// @access  Public
exports.getProductById = async (req, res) => {
    try {
        const product = await Product.findById(req.params.id);
        if (product) {
            res.json(product);
        } else {
            res.status(404).json({ message: 'Product not found' });
        }
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

// @desc    Get product recommendations for logged-in user
// @route   GET /api/products/recommendations
// @access  Private
exports.getRecommendedProducts = async (req, res) => {
    try {
        const user = await User.findById(req.user._id);
        if (!user || !user.bodyShape || !user.skinTone) {
            // If no preferences, return top rated or new arrivals as fallback
            const fallbackProducts = await Product.find({}).sort({ ratings: -1, createdAt: -1 }); // Removed .limit(10)
            return res.json(fallbackProducts);
        }

        // Simple recommendation: products matching body shape OR skin tone tags
        // More advanced: use $and for both, or score based on matches
        const recommendedProducts = await Product.find({
            $or: [
                { bodyShapeTags: user.bodyShape },
                { skinToneTags: user.skinTone }
            ]
        }); // Removed .limit(10)

        res.json(recommendedProducts);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

// @desc    Search products (autocomplete)
// @route   GET /api/products/search/autocomplete
// @access  Public
exports.searchAutocomplete = async (req, res) => {
    const query = req.query.q;
    if (!query) {
        return res.json([]);
    }
    try {
        const products = await Product.find({
            name: { $regex: query, $options: 'i' }
        }).select('name _id category').limit(5); // Send limited fields for autocomplete
        res.json(products);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};