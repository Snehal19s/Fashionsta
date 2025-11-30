// routes/recommendationRoute.js
const express = require('express');
const router  = express.Router();
const Product = require('../models/Product');

/**
 * @route   GET /api/products/recommendations
 * @query   colors=Coral,Lavender&styles=Bohemian,Chic&fits=Slim Fit,Tailored
 * @desc    Return all products matching any of the given color/style/fit tags
 * @access  Public
 */
router.get('/recommendations', async (req, res) => {
  try {
    const { colors = '', styles = '', fits = '' } = req.query;
    const colorList = colors.split(',').map(s=>s.trim()).filter(Boolean);
    const styleList = styles.split(',').map(s=>s.trim()).filter(Boolean);
    const fitList   = fits.split(',').map(s=>s.trim()).filter(Boolean);

    // Require at least one filter
    if (!colorList.length && !styleList.length && !fitList.length) {
      return res
        .status(400)
        .json({ message: 'Provide at least one of colors, styles or fits' });
    }

    // Build OR conditions
    const or = [];
    if (colorList.length) or.push({ colorTags: { $in: colorList } });
    if (styleList.length) or.push({ styleTags: { $in: styleList } });
    if (fitList.length)   or.push({ fitTags:   { $in: fitList   } });

    const products = await Product
      .find({ $or: or })
      .select('-__v')
      .limit(10000);

    res.json(products);
  } catch (err) {
    console.error('Recommendation Error:', err);
    res.status(500).json({ message: 'Server error fetching recommendations' });
  }
});

module.exports = router;
