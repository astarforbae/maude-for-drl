if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <maude_file_path> <checker_path> <maude_path> "
    exit 1
fi
python gen_eval.py --file_path $1 --checker_path $2
$3 evaluate.maude > out.txt
python analyse.py --file_path $1