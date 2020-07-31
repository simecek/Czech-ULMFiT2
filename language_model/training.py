from fastai2.text.all import *
from fastai2.basics import *
import torch


print(torch.cuda.get_device_name(0))

bs=128
data_path = Config.config_path

lang = 'cs'
name = f'{lang}wiki'
path = data_path/name
path.mkdir(exist_ok=True, parents=True)
print(path)

sp = SentencePieceTokenizer(
            lang='cs',
                sp_model=Path('tmp/spm.model'),
                    max_vocab_sz=50000)
tkn = Tokenizer(sp)
print('Setup.........................OK\n')

dls_lm = TextDataLoaders.from_folder(Path("data"), bs=bs, seed=42, 
                                             is_lm=True, tok_tfm=tkn)
print('Data loaders..................OK\n')

learn = language_model_learner(
            dls_lm, AWD_LSTM, drop_mult=0.1, wd=0.1, pretrained=False,
            metrics=[accuracy, Perplexity()])

lr = 3e-3
lr *= bs/48

learn.unfreeze()
learn.fit_one_cycle(10, lr, moms=(0.8, 0.7, 0.8))
print('Training......................OK\n')

learn.save('model/epoch10')
learn.save_encoder('model/epoch10_vocab')
learn.export('model/epoch10.pkl')

learn.summary()

TEXT = "Brno je veliké"
print(learn.predict(TEXT, 5, temperature=1))
