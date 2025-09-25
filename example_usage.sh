#!/bin/bash
#
# Simple Training Examples
# 

echo "🚀 Simple Training Examples"
echo "=========================="
echo

echo "Single GPU training:"
echo "  python train.py configs/simple_nq.json"
echo

echo "Multi-GPU training:"
echo "  torchrun --nproc_per_node=2 train.py configs/simple_nq.json"
echo "  torchrun --nproc_per_node=4 train.py configs/simple_wikitext.json"
echo

echo "Available simple configs:"
ls configs/simple_*.json | sed 's/^/  /'
echo

echo "To create a new config, just create a JSON file like:"
cat << 'EOF'
{
    "description": "My custom dataset training",
    "dataset": "username/my-dataset",
    "split": "train", 
    "text_field": "text",
    "tokens": 10000000
}
EOF
echo

echo "That's it! No complex configurations needed."