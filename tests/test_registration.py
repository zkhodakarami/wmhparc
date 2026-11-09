from wmhparc.registration import TransformChain, ants_apply_command, ants_registration_command


def chain(with_flair=True):
    return TransformChain(template="MNI152NLin2009cAsym", template_image="tpl.nii.gz", anat="t1.nii.gz",
                          anat_modality="T1w", anat_to_template_warp="W.nii.gz",
                          anat_to_template_inverse_warp="IW.nii.gz", anat_to_template_affine="A.mat",
                          flair="flair.nii.gz" if with_flair else None,
                          flair_to_t1_rigid="R.mat" if with_flair else None)


def test_transform_ordering_follows_ants_convention():
    c = chain()
    # subject FLAIR -> template: warp, affine, then the rigid on the moving side
    assert c.to_template("flair") == (["W.nii.gz", "A.mat", "R.mat"], [False, False, False])
    assert c.to_template("anat") == (["W.nii.gz", "A.mat"], [False, False])
    # template atlas -> subject FLAIR: rigid^-1 on the fixed side, affine^-1, inverse warp
    assert c.from_template("flair") == (["R.mat", "A.mat", "IW.nii.gz"], [True, True, False])
    assert c.from_template("anat") == (["A.mat", "IW.nii.gz"], [True, False])


def test_space_of_without_rigid_falls_back_to_anat():
    c = chain(with_flair=False)
    assert c.space_of("FLAIR") == "anat"
    assert c.from_template("flair") == (["A.mat", "IW.nii.gz"], [True, False])


def test_chain_roundtrip(tmp_path):
    c = chain()
    p = c.save(tmp_path / "transforms.json")
    assert TransformChain.load(p) == c
    assert TransformChain.load(tmp_path) == c


def test_cli_commands():
    cmd = ants_registration_command("F.nii.gz", "M.nii.gz", "out/p_", kind="syn", syn_metric="MI")
    assert cmd[0] == "antsRegistration" and "SyN[0.1,3,0]" in cmd
    assert sum(1 for x in cmd if x.startswith("MI[")) == 3 and not any(x.startswith("CC[") for x in cmd)
    rigid = ants_registration_command("F.nii.gz", "M.nii.gz", "p_", kind="rigid")
    assert "Affine[0.1]" not in rigid and "SyN[0.1,3,0]" not in rigid
    app = ants_apply_command("fixed.nii.gz", "atlas.nii.gz", "out.nii.gz", ["R.mat", "A.mat", "IW.nii.gz"],
                             [True, True, False], "label")
    assert app[app.index("-n") + 1] == "GenericLabel[Linear]"
    assert app[app.index("-t") + 1] == "[R.mat,1]" and app[-1] == "IW.nii.gz"
