const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');

const {
  createProduct,
  updateProduct,
  deleteProduct,
  getUsers,
  deleteUser,
  getOrders,
  updateOrderStatus,
} = require('../controllers/adminController');

const { protect } = require('../middlewares/authMiddleware');
const { isAdmin } = require('../middlewares/adminMiddleware');

// ✅ Multer setup for local file storage
const storage = multer.diskStorage({
  destination(req, file, cb) {
    cb(null, path.join(__dirname, '../uploads')); // local folder
  },
  filename(req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  fileFilter(req, file, cb) {
    const allowedTypes = /jpeg|jpg|png/;
    const ext = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mime = allowedTypes.test(file.mimetype);
    if (ext && mime) {
      return cb(null, true);
    } else {
      cb('Only image files (jpg, jpeg, png) are allowed');
    }
  },
});

router.use(protect, isAdmin);

router.post('/products', upload.array('images', 5), createProduct);
router
  .route('/products/:id')
  .put(upload.array('images', 5), updateProduct)
  .delete(deleteProduct);

router.get('/users', getUsers);
router.delete('/users/:id', deleteUser);

router.get('/orders', getOrders);
router.put('/orders/:id/status', updateOrderStatus);

module.exports = router;
