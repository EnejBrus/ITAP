import json
from pathlib import Path
import asyncio
import nbformat
from nbclient import NotebookClient

p = Path('vaje-1-uvod.ipynb')
n = json.loads(p.read_text(encoding='utf-8'))
resitve = {
16: '''# 1.1
imena_vhodov = ["X1", "X2", "X3", "X4", "X5", "X6"]
manjkajo = np.column_stack([np.isnan(X_num[:, :3]), x4 == "", np.isnan(X_num[:, 3:])])
print("Število primerov:", len(X_num), "Število vhodnih spremenljivk:", len(imena_vhodov))
for ime, st, delez in zip(imena_vhodov, manjkajo.sum(axis=0), manjkajo.mean(axis=0)):
    print(f"{ime}: {st} manjkajočih ({delez:.1%})")
print("Y:", np.isnan(y_podatki).sum(), "manjkajočih")
''',
17: '''# 1.2
obdrzi_num = np.isnan(X_num).mean(axis=0) <= 0.20
X_num_izbrani = X_num[:, obdrzi_num]
povprecja = np.nanmean(X_num_izbrani, axis=0)
X_num_cisti = np.where(np.isnan(X_num_izbrani), povprecja, X_num_izbrani)
obdrzi_x4 = (x4 == "").mean() <= 0.20
if obdrzi_x4:
    vrednosti, stevila = np.unique(x4[x4 != ""], return_counts=True)
    moda_x4 = vrednosti[np.argmax(stevila)]
    x4_cisti = np.where(x4 == "", moda_x4, x4)
    print("Moda X4:", moda_x4)
print("Odstranjeni stolpci:", np.array(imena_num)[~obdrzi_num])
''',
18: '''# 1.3
if obdrzi_x4:
    kategorije = np.unique(x4_cisti)
    X_one_hot = (x4_cisti[:, None] == kategorije[None, :]).astype(float)
    imena_one_hot = [f"X4_{v}" for v in kategorije]
    assert np.all(X_one_hot.sum(axis=1) == 1)
else:
    X_one_hot = np.empty((len(X_num), 0))
    imena_one_hot = []
print(imena_one_hot)
print(X_one_hot[:5])
''',
19: '''# 1.4
X_koncna = np.column_stack([X_num_cisti, X_one_hot])
assert np.isfinite(X_koncna).all()
print("Oblika končne tabele:", X_koncna.shape)
print("Manjkajoče vrednosti:", np.isnan(X_koncna).sum())
''',
27: '''# 2.1
def k_kratna_razdelitev(n, K, rng):
    if not 2 <= K <= n:
        raise ValueError("Veljati mora 2 <= K <= n.")
    deli = np.array_split(rng.permutation(n), K)
    return [(np.concatenate(deli[:i] + deli[i + 1:]), iv) for i, iv in enumerate(deli)]

delitve = k_kratna_razdelitev(len(y), 5, np.random.default_rng(42))
vsi_val = np.concatenate([iv for _, iv in delitve])
assert np.array_equal(np.sort(vsi_val), np.arange(len(y)))
assert len(np.unique(vsi_val)) == len(y)
for iu, iv in delitve:
    assert np.intersect1d(iu, iv).size == 0
    assert np.array_equal(np.sort(np.concatenate([iu, iv])), np.arange(len(y)))
print("Velikosti delov:", [len(iv) for _, iv in delitve])
print("Validacijski deli so disjunktni in pokrijejo vse indekse.")
''',
28: '''# 2.2: izbor stolpcev in imputacijo naučimo le na učni množici.
def nauci_imputacijo(X_num, x4):
    obdrzi = np.isnan(X_num).mean(axis=0) <= 0.20
    povprecja = np.nanmean(X_num[:, obdrzi], axis=0)
    obdrzi_kat = (x4 == "").mean() <= 0.20
    moda = None
    if obdrzi_kat:
        vrednosti, stevila = np.unique(x4[x4 != ""], return_counts=True)
        moda = vrednosti[np.argmax(stevila)]
    return {"obdrzi_num": obdrzi, "povprecja": povprecja,
            "obdrzi_x4": obdrzi_kat, "moda_x4": moda}

def imputiraj(X_num, x4, parametri):
    num = X_num[:, parametri["obdrzi_num"]]
    num = np.where(np.isnan(num), parametri["povprecja"], num)
    kat = (np.where(x4 == "", parametri["moda_x4"], x4)
           if parametri["obdrzi_x4"] else None)
    return num, kat

X_num, imena_num, x4, y_podatki = nalozi_podatki_csv()
i_ucni_podatki, _, i_test_podatki = razdeli_indekse(
    len(y_podatki), (0.8, 0.0, 0.2), np.random.default_rng(42))
parametri_imputacije = nauci_imputacijo(X_num[i_ucni_podatki], x4[i_ucni_podatki])
num_ucni, kat_ucni = imputiraj(X_num[i_ucni_podatki], x4[i_ucni_podatki], parametri_imputacije)
num_test, kat_test = imputiraj(X_num[i_test_podatki], x4[i_test_podatki], parametri_imputacije)
assert np.isfinite(num_ucni).all() and np.isfinite(num_test).all()
if kat_ucni is not None:
    assert np.all(kat_ucni != "") and np.all(kat_test != "")
print("Učna in testna tabela:", num_ucni.shape, num_test.shape)
print("Učna povprečja:", parametri_imputacije["povprecja"])
print("Učna moda X4:", parametri_imputacije["moda_x4"])
''',
42: '''# 3.1
def precno_preverjanje(X, y, k, naloga="regresija", K=5, rng=None):
    if naloga not in ("regresija", "klasifikacija"):
        raise ValueError("Neznana naloga.")
    rng = np.random.default_rng() if rng is None else rng
    napake = []
    for iu, iv in k_kratna_razdelitev(len(y), K, rng):
        if not 1 <= k <= len(iu):
            raise ValueError("Neveljaven k.")
        mu_del, sigma_del = nauci_standardizacijo(X[iu])
        sigma_del = np.where(sigma_del == 0, 1.0, sigma_del)
        Zu = standardiziraj(X[iu], mu_del, sigma_del)
        Zv = standardiziraj(X[iv], mu_del, sigma_del)
        napoved = knn_napovej(Zu, y[iu], Zv, k, naloga)
        napaka = mse if naloga == "regresija" else napaka_01
        napake.append(napaka(y[iv], napoved))
    return float(np.mean(napake))
''',
43: '''# 3.2: isto seme za vse k pomeni iste CV delitve.
ks_cv = np.arange(1, 52, 2)
cv_reg = np.array([precno_preverjanje(X[i_tv], y_tv, k, "regresija", 5,
    np.random.default_rng(42)) for k in ks_cv])
cv_clf = np.array([precno_preverjanje(X[i_tv], yc_tv, k, "klasifikacija", 5,
    np.random.default_rng(42)) for k in ks_cv])
k_reg_cv = int(ks_cv[np.argmin(cv_reg)])
k_clf_cv = int(ks_cv[np.argmin(cv_clf)])
print(f"Regresija: k z validacijo = {k_reg}, k s CV = {k_reg_cv}")
print(f"Klasifikacija: k z validacijo = {k_clf}, k s CV = {k_clf_cv}")
print(f"CV: najboljši MSE = {cv_reg.min():.2f}, napaka 0–1 = {cv_clf.min():.3f}")
mu_cv, sigma_cv = nauci_standardizacijo(X[i_tv])
sigma_cv = np.where(sigma_cv == 0, 1.0, sigma_cv)
Zu_cv = standardiziraj(X[i_tv], mu_cv, sigma_cv)
Zte_cv = standardiziraj(X_test, mu_cv, sigma_cv)
napoved_reg_cv = knn_napovej(Zu_cv, y_tv, Zte_cv, k_reg_cv)
napoved_clf_cv = knn_napovej(Zu_cv, yc_tv, Zte_cv, k_clf_cv, "klasifikacija")
beta_cv = nauci_linearno_regresijo(Zu_cv, y_tv)
napoved_lin_cv = napovej_linearno(Zte_cv, beta_cv)
for ime, napoved in [("kNN regresija", napoved_reg_cv),
                      ("Linearna regresija", napoved_lin_cv),
                      ("Osnovni model", np.full(len(y_test), y_tv.mean()))]:
    print(f"{ime}: MSE = {mse(y_test, napoved):.2f}, RMSE = {np.sqrt(mse(y_test, napoved)):.2f}")
print(f"kNN klasifikacija: napaka = {napaka_01(yc_test, napoved_clf_cv):.3f}")
print(f"Osnovni klasifikator: napaka = {napaka_01(yc_test, np.full(len(yc_test), np.bincount(yc_tv).argmax())):.3f}")
# Testnih rezultatov ne uporabljamo za prilagajanje modelov.
''',
44: '''# 3.3
napovedi_33 = {"kNN klasifikator": napoved_clf_cv,
    "kNN regresija s pragom": (napoved_reg_cv >= prag).astype(int),
    "Linearna regresija s pragom": (napoved_lin_cv >= prag).astype(int)}
napake_33 = {ime: napaka_01(yc_test, napoved) for ime, napoved in napovedi_33.items()}
for ime, napaka in napake_33.items():
    print(f"{ime}: napaka 0–1 = {napaka:.3f}")
print("Najnižja napaka na tej testni množici:",
      ", ".join(ime for ime, napaka in napake_33.items() if napaka == min(napake_33.values())))
''',
45: '''# 3.4: vrstice so dejanski, stolpci napovedani razredi (0, 1).
def metrike_klasifikacije(y_pravi, y_napoved):
    matrika = np.bincount(2 * y_pravi.astype(int) + y_napoved.astype(int),
                          minlength=4).reshape(2, 2)
    tn, fp, fn, tp = matrika.ravel()
    preciznost = tp / (tp + fp) if tp + fp else np.nan
    priklic = tp / (tp + fn) if tp + fn else np.nan
    return matrika, preciznost, priklic

for ime, napoved in napovedi_33.items():
    matrika, preciznost, priklic = metrike_klasifikacije(yc_test, napoved)
    assert matrika.sum() == len(yc_test)
    assert np.isclose((matrika[0, 1] + matrika[1, 0]) / matrika.sum(), napake_33[ime])
    print(f"\\n{ime}\\n{matrika}")
    print(f"Preciznost = {preciznost:.3f}, priklic = {priklic:.3f}")
'''
}
for i, source in resitve.items():
    n['cells'][i]['source'] = source.splitlines(keepends=True)
    n['cells'][i]['outputs'] = []
    n['cells'][i]['execution_count'] = None
n['metadata']['kernelspec'] = {'display_name': 'Python (ITAP26)', 'language': 'python', 'name': 'itap26'}
razlaga = '''### Razlaga naloge 3

CV izbere k=13 za regresijo in k=51 za klasifikacijo, ena validacijska delitev pa k=9 oziroma k=13. Rezultati se razlikujejo, ker CV povpreči napake več delitev in je manj odvisen od ene sestave validacijskih primerov.

Na tej testni množici ima najnižjo napako 0–1 linearna regresija s pragom (0,281), sledita kNN klasifikator (0,303) in kNN regresija s pragom (0,326). To ni dokaz, da je pristop vedno boljši. Regresija uporabi tudi velikost cilja in lahko dobro deluje, če so napovedi okoli praga natančne. Lahko deluje slabše, ker MSE ni napaka 0–1: oddaljene velike vrednosti močno vplivajo na MSE, povprečenje pa lahko napoved premakne čez prag. Testnih rezultatov ne uporabljamo za novo izbiro hiperparametrov.

Matrika zmede je [[TN, FP], [FN, TP]]. Preciznost TP/(TP+FP) je delež pravilnih pozitivnih napovedi, priklic TP/(TP+FN) pa delež odkritih dejansko pozitivnih primerov. Pri ničelnem imenovalcu je metrika nedefinirana (nan).
'''
if not any('### Razlaga naloge 3' in ''.join(c['source']) for c in n['cells']):
    n['cells'].insert(46, {'cell_type': 'markdown', 'metadata': {}, 'id': 'razlaga-naloge-3',
                            'source': razlaga.splitlines(keepends=True)})
p.write_text(json.dumps(n, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
nb = nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=180, kernel_name='itap26').execute()
nbformat.write(nb, p)
preverjen = nbformat.read(p, as_version=4)
code = [c for c in preverjen.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code)
assert not any('# TODO' in c.source for c in code)
assert not any(o.output_type == 'error' for c in code for o in c.outputs)
print(f'SHRANJENO IN PREVERJENO: {p.resolve()}')
print(f'Izvedenih {len(code)}/{len(code)} programskih celic, brez TODO in brez napak.')
