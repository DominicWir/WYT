#!/bin/bash

# Package files for Google Colab upload
# This script creates a ZIP file with all necessary code files

echo "================================================"
echo "PACKAGING FILES FOR GOOGLE COLAB"
echo "================================================"
echo ""

# Create temporary directory
TEMP_DIR="colab_package"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

echo "1. Copying Python files..."
cp train_activity_colab.py $TEMP_DIR/
cp xlstm_model.py $TEMP_DIR/
cp train.py $TEMP_DIR/
cp data_preprocessing.py $TEMP_DIR/
cp feature_extraction.py $TEMP_DIR/
cp load_zip_datasets.py $TEMP_DIR/
cp inference.py $TEMP_DIR/
cp config.yaml $TEMP_DIR/
cp requirements.txt $TEMP_DIR/

echo "   ✓ Copied 9 files"

echo ""
echo "2. Creating README for Colab..."
cat > $TEMP_DIR/COLAB_README.txt << 'EOF'
GOOGLE COLAB SETUP INSTRUCTIONS
================================

1. Go to: https://colab.research.google.com
2. Create new notebook
3. Enable GPU: Runtime -> Change runtime type -> T4 GPU
4. Upload all files from this package
5. Upload your dataset ZIP file (opportunity+activity+recognition.zip)
6. Run: !python train_activity_colab.py

Expected training time: 1-2 hours

For detailed instructions, see the implementation_plan.md
EOF

echo "   ✓ Created README"

echo ""
echo "3. Creating ZIP package..."
zip -r colab_training_package.zip $TEMP_DIR/
echo "   ✓ Package created: colab_training_package.zip"

echo ""
echo "4. Cleaning up..."
rm -rf $TEMP_DIR
echo "   ✓ Cleanup complete"

echo ""
echo "================================================"
echo "✓ PACKAGING COMPLETE"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Upload 'colab_training_package.zip' to Google Colab"
echo "  2. Unzip in Colab: !unzip colab_training_package.zip"
echo "  3. Upload dataset: opportunity+activity+recognition.zip"
echo "  4. Run training: !python train_activity_colab.py"
echo ""
echo "Package size:"
ls -lh colab_training_package.zip
echo ""
