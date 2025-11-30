const Product = require('../models/Product');
const User = require('../models/User');
const Order = require('../models/Order');

// --- Product Management ---

// @desc    Create a product
// @route   POST /api/admin/products
// @access  Private/Admin
exports.createProduct = async (req, res) => {
  try {
    const imageUrls = req.files.map(f => `/uploads/${f.filename}`);
    const sizes = req.body.sizes?.split(',').map(s => s.trim()) || [];
    const bodyShapeTags = req.body.bodyShapeTags?.split(',').map(t => t.trim()) || [];
    const skinToneTags = req.body.skinToneTags?.split(',').map(t => t.trim()) || [];

    const product = new Product({
      name: req.body.name,
      description: req.body.description,
      price: req.body.price,
      category: req.body.category,
      gender: req.body.gender,
      stock: req.body.stock,
      sizes,
      bodyShapeTags,
      skinToneTags,
      images: imageUrls,
    });

    await product.save();
    res.status(201).json({ message: 'Product created successfully', product });
  } catch (error) {
    res.status(500).json({ message: 'Product creation failed', error: error.message });
  }
};



// @desc    Update a product
// @route   PUT /api/admin/products/:id
// @access  Private/Admin
exports.updateProduct = async (req, res) => {
  try {
    const {
      name,
      category,
      price,
      description,
      gender,
      bodyShapeTags,
      skinToneTags,
      sizes,
      stock,
    } = req.body;

    const product = await Product.findById(req.params.id);
    if (!product) return res.status(404).json({ message: 'Product not found' });

    product.name = name || product.name;
    product.category = category || product.category;
    product.price = price || product.price;
    product.description = description || product.description;
    product.gender = gender || product.gender;
    product.bodyShapeTags = bodyShapeTags ? bodyShapeTags.split(',').map(tag => tag.trim()) : product.bodyShapeTags;
    product.skinToneTags = skinToneTags ? skinToneTags.split(',').map(tag => tag.trim()) : product.skinToneTags;
    product.sizes = sizes ? sizes.split(',').map(s => s.trim()) : product.sizes;
    product.stock = stock !== undefined ? stock : product.stock;

    if (req.files && req.files.length > 0) {
      const newImageUrls = req.files.map(f => `/uploads/${f.filename}`);
      product.images = newImageUrls;
    }

    const updatedProduct = await product.save();
    res.json(updatedProduct);
  } catch (error) {
    console.error("Update Product Error:", error);
    res.status(500).json({ message: error.message });
  }
};


// @desc    Delete a product
// @route   DELETE /api/admin/products/:id
// @access  Private/Admin
exports.deleteProduct = async (req, res) => {
  try {
    const product = await Product.findById(req.params.id);
    if (!product) return res.status(404).json({ message: 'Product not found' });

    await product.deleteOne();
    res.json({ message: 'Product removed' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// --- User Management ---

// @desc    Get all users
// @route   GET /api/admin/users
// @access  Private/Admin
exports.getUsers = async (req, res) => {
  try {
    const users = await User.find({}).select('-password');
    res.json(users);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// @desc    Delete a user
// @route   DELETE /api/admin/users/:id
// @access  Private/Admin
exports.deleteUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) return res.status(404).json({ message: 'User not found' });

    if (user.isAdmin) {
      return res.status(400).json({ message: 'Cannot delete admin user' });
    }

    await user.deleteOne();
    res.json({ message: 'User removed' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// --- Order Management ---

// @desc    Get all orders
// @route   GET /api/admin/orders
// @access  Private/Admin
exports.getOrders = async (req, res) => {
  try {
    const orders = await Order.find({}).populate('userId', 'name email').sort({ date: -1 });
    res.json(orders);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// @desc    Update order status
// @route   PUT /api/admin/orders/:id/status
// @access  Private/Admin
exports.updateOrderStatus = async (req, res) => {
  const { orderStatus, paymentStatus } = req.body;
  try {
    const order = await Order.findById(req.params.id);
    if (!order) return res.status(404).json({ message: 'Order not found' });

    order.orderStatus = orderStatus || order.orderStatus;
    order.paymentStatus = paymentStatus || order.paymentStatus;

    const updatedOrder = await order.save();
    res.json(updatedOrder);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
