path = kagglehub.dataset_download("youssefelbadry10/english-to-arabic")
files = os.listdir(path)
print(files)
txt_file = os.path.join(path, 'ara.txt')
df = pd.read_csv(txt_file, sep='\t', names=['english', 'arabic', 'meta'], encoding= 'utf-8')
df = df.dropna()
print(df.head())
txt_file_out = '/content/ara.txt'
with open(txt_file_out, 'w', encoding='utf-8') as f:
  for line in df['english'].tolist() + df['arabic'].tolist():
    f.write(line + '\n')
