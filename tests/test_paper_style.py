import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foraois.utils import paper_style as ps


def test_labels_avoid_python_exponent_formatting():
    assert ps.m2_label(1.0e12) == r"$M_2=10^{12}\,M_\odot/h$"
    assert ps.m2_label(3.16e13) == r"$M_2=3.16\times10^{13}\,M_\odot/h$"
    assert ps.sci(8.36e11) == r"8.36\times10^{11}"
    assert ps.sci(5e10, 0) == r"5\times10^{10}"


def test_stem_and_save_write_pdf_and_png(tmp_path):
    assert ps.stem_of("a/b.png").as_posix() == "a/b"
    assert ps.stem_of("a/b").as_posix() == "a/b"
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COL, 2.5))
    ax.plot([0, 1], [0, 1])
    ps.save(fig, tmp_path / "fig.png")
    assert (tmp_path / "fig.pdf").exists() and (tmp_path / "fig.png").exists()
    plt.close(fig)


def test_data_round_trip(tmp_path):
    ps.save_data(tmp_path / "x.png", {"a": [1, 2, 3]})
    assert ps.load_data(tmp_path / "x.pkl") == {"a": [1, 2, 3]}
