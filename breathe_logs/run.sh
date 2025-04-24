for file in logs/accuracy/*.txt; do
  name=$(basename $file .txt)
  python parse_plot.py $name
  python process_signals.py $name
done
