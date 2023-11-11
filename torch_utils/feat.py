"""
This program is a copy and modification of deepchem.utils.molecule_feature_utils.py under the MIT License.

Original source code:
- https://github.com/deepchem/deepchem/blob/0611ac54e66589956435a7ea30a91f80b49d88d5/deepchem/utils/molecule_feature_utils.py
- https://github.com/deepchem/deepchem/blob/0df62b6e6315509dd1c7c0ef8b6caf5f253b519b/deepchem/feat/molecule_featurizers/mol_graph_conv_featurizer.py

License:
- https://github.com/deepchem/deepchem/blob/0611ac54e66589956435a7ea30a91f80b49d88d5/LICENSE


Utilities for constructing node features or bond features.
Some functions are based on chainer-chemistry or dgl-lifesci.

Repositories:
- https://github.com/chainer/chainer-chemistry
- https://github.com/awslabs/dgl-lifesci
"""  # noqa: E501

from typing import Optional
import os
import logging
from typing import List, Union, Tuple

import numpy as np
import rdkit.Chem.rdchem
from rdkit import Chem

import torch
import torch_geometric.data

logger = logging.getLogger(__name__)

DEFAULT_ATOM_TYPE_SET = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_FORMAL_CHARGE_SET = [-2, -1, 0, 1, 2]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_RING_SIZE_SET = [3, 4, 5, 6, 7, 8]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
DEFAULT_GRAPH_DISTANCE_SET = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_ATOM_IMPLICIT_VALENCE_SET = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_ATOM_EXPLICIT_VALENCE_SET = [1, 2, 3, 4, 5, 6]
ALLEN_ELECTRONEGATIVTY = {  # Allen scale electronegativity
    "H": 2.3,
    "He": 4.160,
    "Li": 0.912,
    "Be": 1.576,
    "B": 2.051,
    "C": 2.544,
    "N": 3.066,
    "O": 3.610,
    "F": 4.193,
    "Ne": 4.787,
    "Na": 0.869,
    "Mg": 1.293,
    "Al": 1.613,
    "Si": 1.916,
    "P": 2.253,
    "S": 2.589,
    "Cl": 2.869,
    "Ar": 3.242,
    "K": 0.734,
    "Ca": 1.034,
    "Sc": 1.19,
    "Ti": 1.38,
    "V": 1.53,
    "Cr": 1.65,
    "Mn": 1.75,
    "Fe": 1.80,
    "Co": 1.84,
    "Ni": 1.88,
    "Cu": 1.85,
    "Zn": 1.588,
    "Ga": 1.756,
    "Ge": 1.994,
    "As": 2.211,
    "Se": 2.424,
    "Br": 2.685,
    "Kr": 2.966,
    "Rb": 0.706,
    "Sr": 0.963,
    "Y": 1.12,
    "Zr": 1.32,
    "Nb": 1.41,
    "Mo": 1.47,
    "Tc": 1.51,
    "Ru": 1.54,
    "Rh": 1.56,
    "Pd": 1.58,
    "Ag": 1.87,
    "Cd": 1.521,
    "In": 1.656,
    "Sn": 1.824,
    "Sb": 1.984,
    "Te": 2.158,
    "I": 2.359,
    "Xe": 2.582,
    "Cs": 0.659,
    "Ba": 0.881,
    "Lu": 1.09,
    "Hf": 1.16,
    "Ta": 1.34,
    "W": 1.47,
    "Re": 1.60,
    "Os": 1.65,
    "Ir": 1.68,
    "Pt": 1.72,
    "Au": 1.92,
    "Hg": 1.765,
    "Tl": 1.789,
    "Pb": 1.854,
    "Bi": 2.01,
    "Po": 2.19,
    "At": 2.39,
    "Rn": 2.60,
    "Fr": 0.67,
    "Ra": 0.89,
}


def _construct_atom_feature(
    atom: rdkit.Chem.rdchem.Atom,
    h_bond_infos: List[Tuple[int, str]],
    use_chirality: bool,
    use_partial_charge: bool,
) -> np.ndarray:
    """Construct an atom feature from a RDKit atom object.

    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    h_bond_infos: List[Tuple[int, str]]
        A list of tuple `(atom_index, hydrogen_bonding_type)`.
        Basically, it is expected that this value is the return value of
        `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
        value is "Acceptor" or "Donor".
    use_chirality: bool
        Whether to use chirality information or not.
    use_partial_charge: bool
        Whether to use partial charge data or not.

    Returns
    -------
    np.ndarray
        A one-hot vector of the atom feature.

    """  # noqa: E501
    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
    atom_feat = np.concatenate(
        [
            atom_type,
            formal_charge,
            hybridization,
            acceptor_donor,
            aromatic,
            degree,
            total_num_Hs,
        ]
    )

    if use_chirality:
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, np.array(chirality)])

    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])
    return atom_feat


def _construct_bond_feature(bond: rdkit.Chem.rdchem.Bond) -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    np.ndarray
        A one-hot vector of the bond feature.

    """  # noqa: E501
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


def mol2data(
    mol: rdkit.Chem.rdchem.Mol,
    y: Optional[torch.Tensor] = None,
    use_chirality: bool = False,
    use_partial_charge: bool = False,
    use_edges: bool = False,
) -> torch_geometric.data.Data:
    """Calculate molecule graph features from RDKit mol object.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        RDKit mol object.
    y: Optional[torch.Tensor], default None
        Target value.
    use_chirality: bool, default False
        Whether to use chirality information or not.
    use_partial_charge: bool, default False
        Whether to use partial charge data or not.
    use_edges: bool, default False
        Whether to use edge features or not.


    Returns
    -------
    graph: GraphData
        A molecule graph with some features.

    """  # noqa: E501
    if mol.GetNumAtoms() == 0:
        raise ValueError(
            "More than one atom should be present"
            " in the molecule for this featurizer to work. ({})".format(
                Chem.MolToSmiles(mol)
            )  # noqa: E501
        )

    if use_partial_charge:
        try:
            mol.GetAtomWithIdx(0).GetProp("_GasteigerCharge")
        except Exception:  # HACK: should be more specific
            # If partial charges were not computed
            try:
                from rdkit.Chem import AllChem

                AllChem.ComputeGasteigerCharges(mol)
            except ModuleNotFoundError:
                raise ImportError("This class requires RDKit to be installed.")

    # construct atom (node) feature
    h_bond_infos = construct_hydrogen_bonding_info(mol)
    atom_features = np.asarray(
        [
            _construct_atom_feature(
                atom,
                h_bond_infos,
                use_chirality,
                use_partial_charge,
            )
            for atom in mol.GetAtoms()
        ],
        dtype=float,
    )

    # construct edge (bond) index
    src, dest = [], []
    for bond in mol.GetBonds():
        # add edge list considering a directed graph
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [start, end]
        dest += [end, start]

    # construct edge (bond) feature
    bond_features = None  # deafult None
    if use_edges:
        features = []
        for bond in mol.GetBonds():
            features += 2 * [_construct_bond_feature(bond)]
        bond_features = np.asarray(features, dtype=float)

    # HACK: Need pos or not in 'Data'?

    # load_sdf_files returns pos as strings but user can also specify
    # numpy arrays for atom coordinates
    # pos = []
    # if "pos_x" in kwargs and "pos_y" in kwargs and "pos_z" in kwargs:
    #     if isinstance(kwargs["pos_x"], str):
    #         pos_x = eval(kwargs["pos_x"])
    #     elif isinstance(kwargs["pos_x"], np.ndarray):
    #         pos_x = kwargs["pos_x"]
    #     if isinstance(kwargs["pos_y"], str):
    #         pos_y = eval(kwargs["pos_y"])
    #     elif isinstance(kwargs["pos_y"], np.ndarray):
    #         pos_y = kwargs["pos_y"]
    #     if isinstance(kwargs["pos_z"], str):
    #         pos_z = eval(kwargs["pos_z"])
    #     elif isinstance(kwargs["pos_z"], np.ndarray):
    #         pos_z = kwargs["pos_z"]

    #     for x, y, z in zip(pos_x, pos_y, pos_z):
    #         pos.append([x, y, z])
    #     node_pos_features = np.asarray(pos)
    # else:
    #     node_pos_features = None
    # return GraphData(node_features=atom_features,
    #                      edge_index=np.asarray([src, dest], dtype=int),
    #                      edge_features=bond_features,
    #                      node_pos_features=node_pos_features)

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor([src, dest], dtype=torch.long)
    edge_attr = (
        torch.tensor(bond_features, dtype=torch.float) if use_edges else None
    )
    y = torch.tensor([y], dtype=torch.float) if y is not None else None

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )


class _ChemicalFeaturesFactory:
    """This is a singleton class for RDKit base features."""  # noqa: E501

    _instance = None

    @classmethod
    def get_instance(cls):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if not cls._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
        return cls._instance


def one_hot_encode(
    val: Union[int, str],
    allowable_set: Union[List[str], List[int]],
    include_unknown_set: bool = False,
) -> List[float]:
    """One hot encoder for elements of a provided set.

    Examples
    --------
    >>> one_hot_encode("a", ["a", "b", "c"])
    [1.0, 0.0, 0.0]
    >>> one_hot_encode(2, [0, 1, 2])
    [0.0, 0.0, 1.0]
    >>> one_hot_encode(3, [0, 1, 2])
    [0.0, 0.0, 0.0]
    >>> one_hot_encode(3, [0, 1, 2], True)
    [0.0, 0.0, 0.0, 1.0]

    Parameters
    ----------
    val: int or str
        The value must be present in `allowable_set`.
    allowable_set: List[int] or List[str]
        List of allowable quantities.
    include_unknown_set: bool, default False
        If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        An one-hot vector of val.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    Raises
    ------
    ValueError
        If include_unknown_set is False and `val` is not in `allowable_set`.
    """  # noqa: E501
    if include_unknown_set is False:
        if val not in allowable_set:
            logger.info(
                "input {0} not in allowable set {1}:".format(
                    val, allowable_set
                )
            )

    # init an one-hot vector
    if include_unknown_set is False:
        one_hot_legnth = len(allowable_set)
    else:
        one_hot_legnth = len(allowable_set) + 1
    one_hot = [0.0 for _ in range(one_hot_legnth)]

    try:
        one_hot[allowable_set.index(val)] = 1.0  # type: ignore
    except Exception:  # HACK: should be certain exception
        if include_unknown_set:
            # If include_unknown_set is True, set the last index is 1.
            one_hot[-1] = 1.0
        else:
            pass
    return one_hot


#################################################################
# atom (node) featurization
#################################################################


def get_atom_type_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
    allowable_set: List[str] = DEFAULT_ATOM_TYPE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of an atom type.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[str]
        The atom types to consider. The default set is
        `["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]`.
    include_unknown_set: bool, default True
        If true, the index of all atom not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        An one-hot vector of atom types.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """  # noqa: E501
    return one_hot_encode(atom.GetSymbol(), allowable_set, include_unknown_set)


def construct_hydrogen_bonding_info(
    mol: rdkit.Chem.rdchem.Mol,
) -> List[Tuple[int, str]]:
    """Construct hydrogen bonding infos about a molecule.

    Parameters
    ---------
    mol: rdkit.Chem.rdchem.Mol
        RDKit mol object

    Returns
    -------
    List[Tuple[int, str]]
        A list of tuple `(atom_index, hydrogen_bonding_type)`.
        The `hydrogen_bonding_type` value is "Acceptor" or "Donor".
    """  # noqa: E501
    factory = _ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding = []
    for f in feats:
        hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))
    return hydrogen_bonding


def get_atom_hydrogen_bonding_one_hot(
    atom: rdkit.Chem.rdchem.Atom, hydrogen_bonding: List[Tuple[int, str]]
) -> List[float]:
    """Get an one-hot feat about whether an atom accepts electrons or donates electrons.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    hydrogen_bonding: List[Tuple[int, str]]
        The return value of `construct_hydrogen_bonding_info`.
        The value is a list of tuple `(atom_index, hydrogen_bonding)` like (1, "Acceptor").

    Returns
    -------
    List[float]
        A one-hot vector of the ring size type. The first element
        indicates "Donor", and the second element indicates "Acceptor".
    """  # noqa: E501
    one_hot = [0.0, 0.0]
    atom_idx = atom.GetIdx()
    for hydrogen_bonding_tuple in hydrogen_bonding:
        if hydrogen_bonding_tuple[0] == atom_idx:
            if hydrogen_bonding_tuple[1] == "Donor":
                one_hot[0] = 1.0
            elif hydrogen_bonding_tuple[1] == "Acceptor":
                one_hot[1] = 1.0
    return one_hot


def get_atom_is_in_aromatic_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
) -> List[float]:
    """Get ans one-hot feature about whether an atom is in aromatic system or not.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A vector of whether an atom is in aromatic system or not.
    """  # noqa: E501
    return [float(atom.GetIsAromatic())]


def get_atom_hybridization_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
    allowable_set: List[str] = DEFAULT_HYBRIDIZATION_SET,
    include_unknown_set: bool = False,
) -> List[float]:
    """Get an one-hot feature of hybridization type.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[str]
        The hybridization types to consider. The default set is `["SP", "SP2", "SP3"]`
    include_unknown_set: bool, default False
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        An one-hot vector of the hybridization type.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """  # noqa: E501
    return one_hot_encode(
        str(atom.GetHybridization()), allowable_set, include_unknown_set
    )


def get_atom_total_num_Hs_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
    allowable_set: List[int] = DEFAULT_TOTAL_NUM_Hs_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of the number of hydrogens which an atom has.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        The number of hydrogens to consider. The default set is `[0, 1, ..., 4]`
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the number of hydrogens which an atom has.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """  # noqa: E501
    return one_hot_encode(
        atom.GetTotalNumHs(), allowable_set, include_unknown_set
    )


def get_atom_chirality_one_hot(atom: rdkit.Chem.rdchem.Atom) -> List[float]:
    """Get an one-hot feature about an atom chirality type.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A one-hot vector of the chirality type. The first element
        indicates "R", and the second element indicates "S".
    """  # noqa: E501
    one_hot = [0.0, 0.0]
    try:
        chiral_type = atom.GetProp("_CIPCode")
        if chiral_type == "R":
            one_hot[0] = 1.0
        elif chiral_type == "S":
            one_hot[1] = 1.0
    except Exception:  # HACK: should be ce
        pass
    return one_hot


def get_atom_formal_charge(atom: rdkit.Chem.rdchem.Atom) -> List[float]:
    """Get a formal charge of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A vector of the formal charge.
    """  # noqa: E501
    return [float(atom.GetFormalCharge())]


def get_atom_formal_charge_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
    allowable_set: List[int] = DEFAULT_FORMAL_CHARGE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get one hot encoding of formal charge of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        The degree to consider. The default set is `[-2, -1, ..., 2]`
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.


    Returns
    -------
    List[float]
        A vector of the formal charge.
    """  # noqa: E501
    return one_hot_encode(
        atom.GetFormalCharge(), allowable_set, include_unknown_set
    )


def get_atom_partial_charge(atom: rdkit.Chem.rdchem.Atom) -> List[float]:
    """Get a partial charge of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A vector of the parital charge.

    Notes
    -----
    Before using this function, you must calculate `GasteigerCharge`
    like `AllChem.ComputeGasteigerCharges(mol)`.
    """  # noqa: E501
    gasteiger_charge = atom.GetProp("_GasteigerCharge")
    if gasteiger_charge in ["-nan", "nan", "-inf", "inf"]:
        gasteiger_charge = 0.0
    return [float(gasteiger_charge)]


def get_atom_total_degree_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
    allowable_set: List[int] = DEFAULT_TOTAL_DEGREE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of the degree which an atom has.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        The degree to consider. The default set is `[0, 1, ..., 5]`
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the degree which an atom has.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """  # noqa: E501
    return one_hot_encode(
        atom.GetTotalDegree(), allowable_set, include_unknown_set
    )


def get_atom_implicit_valence_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
    allowable_set: List[int] = DEFAULT_ATOM_IMPLICIT_VALENCE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of implicit valence of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        Atom implicit valence to consider. The default set is `[0, 1, ..., 6]`
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of implicit valence an atom has.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    """  # noqa: E501
    return one_hot_encode(
        atom.GetImplicitValence(), allowable_set, include_unknown_set
    )


def get_atom_explicit_valence_one_hot(
    atom: rdkit.Chem.rdchem.Atom,
    allowable_set: List[int] = DEFAULT_ATOM_EXPLICIT_VALENCE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of explicit valence of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        Atom explicit valence to consider. The default set is `[1, ..., 6]`
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of explicit valence an atom has.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    """  # noqa: E501
    return one_hot_encode(
        atom.GetExplicitValence(), allowable_set, include_unknown_set
    )


#################################################################
# bond (edge) featurization
#################################################################


def get_bond_type_one_hot(
    bond: rdkit.Chem.rdchem.Bond,
    allowable_set: List[str] = DEFAULT_BOND_TYPE_SET,
    include_unknown_set: bool = False,
) -> List[float]:
    """Get an one-hot feature of bond type.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object
    allowable_set: List[str]
        The bond types to consider. The default set is `["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]`.
    include_unknown_set: bool, default False
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the bond type.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """  # noqa: E501
    return one_hot_encode(
        str(bond.GetBondType()), allowable_set, include_unknown_set
    )


def get_bond_is_in_same_ring_one_hot(
    bond: rdkit.Chem.rdchem.Bond,
) -> List[float]:
    """Get an one-hot feature about whether atoms of a bond is in the same ring or not.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    List[float]
        A one-hot vector of whether a bond is in the same ring or not.
    """  # noqa: E501
    return [int(bond.IsInRing())]


def get_bond_is_conjugated_one_hot(
    bond: rdkit.Chem.rdchem.Bond,
) -> List[float]:
    """Get an one-hot feature about whether a bond is conjugated or not.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    List[float]
        A one-hot vector of whether a bond is conjugated or not.
    """  # noqa: E501
    return [int(bond.GetIsConjugated())]


def get_bond_stereo_one_hot(
    bond: rdkit.Chem.rdchem.Bond,
    allowable_set: List[str] = DEFAULT_BOND_STEREO_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of the stereo configuration of a bond.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object
    allowable_set: List[str]
        The stereo configuration types to consider.
        The default set is `["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]`.
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the stereo configuration of a bond.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """  # noqa: E501
    return one_hot_encode(
        str(bond.GetStereo()), allowable_set, include_unknown_set
    )


def get_bond_graph_distance_one_hot(
    bond: rdkit.Chem.rdchem.Bond,
    graph_dist_matrix: np.ndarray,
    allowable_set: List[int] = DEFAULT_GRAPH_DISTANCE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of graph distance.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object
    graph_dist_matrix: np.ndarray
        The return value of `Chem.GetDistanceMatrix(mol)`. The shape is `(num_atoms, num_atoms)`.
    allowable_set: List[int]
        The graph distance types to consider. The default set is `[1, 2, ..., 7]`.
    include_unknown_set: bool, default False
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the graph distance.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """  # noqa: E501
    graph_dist = graph_dist_matrix[
        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    ]
    return one_hot_encode(graph_dist, allowable_set, include_unknown_set)
