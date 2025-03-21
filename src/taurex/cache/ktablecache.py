"""Contains caching class for Molecular ktable files."""

import typing as t

from taurex.log import Logger

from ..opacity.ktables import KTable
from . import GlobalCache
from .singleton import Singleton


class KTableCache(Singleton):
    """Implements a lazy load of opacities.

    A singleton that loads and caches ktables as they are needed.

    Now TauREx3 will use it instead in all calculations!

    """

    def init(self) -> None:
        """Initialise the cache."""
        self.opacity_dict: t.Dict[str, KTable] = {}
        self._opacity_path = GlobalCache()["ktable_path"]
        self.log = Logger("KTableCache")
        self._default_interpolation = "linear"
        self._memory_mode = True

    def set_ktable_path(self, opacity_path: str) -> None:
        """Set the path(s) that will be searched for ktables.

        Opacities in this path must be of supported types:

            - HDF5 opacities
            - ``.pickle`` opacities
            - ExoTransmit opacities.

        Parameters
        ----------

        opacity_path : str or :obj:`list` of str, optional
            search path(s) to look for molecular opacities

        """

        import os

        GlobalCache()["ktable_path"] = opacity_path

        if not os.path.isdir(opacity_path):
            self.log.error("PATH: %s does not exist!!!", opacity_path)
            raise NotADirectoryError
        self.log.debug("Path set to %s", opacity_path)

    def __getitem__(self, key: str) -> KTable:
        """For a molecule return the relevant opacity.

        Opacity return is an :class:`~taurex.opacity.opacity.Opacity` object.


        Parameter
        ---------
        key : str
            molecule name

        Returns
        -------
        :class:`~taurex.opacity.pickleopacity.PickleOpacity`
            Cross-section object desired

        Raises
        ------
        Exception
            If molecule could not be loaded/found

        """
        if key in self.opacity_dict:
            return self.opacity_dict[key]
        else:
            # Try a load of the opacity
            self.load_opacity(molecule_filter=[key])
            # If we have it after a load then good job boys
            if key in self.opacity_dict:
                return self.opacity_dict[key]
            else:
                # try:
                #     # if self._radis:
                #     #     return self.create_radis_opacity(key,molecule_filter=[key])
                #     # else:
                #         raise Exception
                # except Exception as e:
                # Otherwise throw an error
                self.log.error("Opacity for molecule %s could not be loaded", key)
                self.log.error(
                    "It could not be found in the local dictionary %s",
                    list(self.opacity_dict.keys()),
                )
                self.log.error("Or paths %s", self._opacity_path)
                self.log.error("Try loading it manually/ putting it in a path")
                raise Exception("Opacity could not be loaded")

    def add_opacity(
        self, opacity: KTable, molecule_filter: t.Optional[t.List[str]] = None
    ) -> None:
        """Adds a :class:`~taurex.opacity.ktable.KTable` object to the cache

        Parameters
        ----------
        opacity : :class:`~taurex.opacity.ktable.KTable`
            Opacity object to add to the cache

        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity object will only be included
            if its molecule is in the list. Mostly used by the
            :func:`__getitem__` for filtering

        """
        self.log.info("Reading opacity %s", opacity.moleculeName)
        if opacity.moleculeName in self.opacity_dict:
            self.log.warning(
                "Opacity with name %s already in opactiy dictionary %s skipping",
                opacity.moleculeName,
                self.opacity_dict.keys(),
            )
            return
        if molecule_filter is not None:
            if opacity.moleculeName in molecule_filter:
                self.log.info("Loading opacity %s into model", opacity.moleculeName)
                self.opacity_dict[opacity.moleculeName] = opacity
        else:
            self.log.info("Loading opacity %s into model", opacity.moleculeName)
            self.opacity_dict[opacity.moleculeName] = opacity

    def find_list_of_molecules(self) -> t.Set[str]:
        """Returns a list of molecules that can be loaded.

        Returns
        -------
        :obj:`set` of str
            List of molecules that can be loaded

        """
        from taurex.parameter.classfactory import ClassFactory

        opacity_klasses = ClassFactory().ktableKlasses

        molecules = []

        for c in opacity_klasses:
            molecules.extend([x[0] for x in c.discover()])

        forced = []

        return set(molecules + forced)

    def load_opacity_from_path(
        self, path: str, molecule_filter: t.Optional[t.List[str]] = None
    ) -> None:
        """Searches path for and loads molecular cross-section files.

        For example, if the path is ``/path/to/opacities`` then
        ``/path/to/opacities/H2O.pickle`` will be loaded as
        :class:`~taurex.opacity.pickleopacity.PickleOpacity`


        Parameters
        ----------
        path : str
            Path to search for molecular cross-section files

        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity will only be loaded
            if its molecule is in this list. Mostly used by the
            :func:`__getitem__` for filtering

        """

        from taurex.parameter.classfactory import ClassFactory

        cf = ClassFactory()

        opacity_klass_list = sorted(cf.ktableKlasses, key=lambda x: x.priority())
        for c in opacity_klass_list:
            try:
                discover = c.discover()
            except NotImplementedError:
                self.log.warning("Klass %s has no discover method", c.__name__)
                continue
            for mol, args in discover:
                self.log.debug("Klass: %s %s", mol, args)
                op = None
                if mol in molecule_filter:
                    if not isinstance(
                        args,
                        (
                            list,
                            tuple,
                        ),
                    ):
                        args = [args]
                    op = c(*args)
                if op is not None and op.moleculeName not in self.opacity_dict:
                    self.add_opacity(op, molecule_filter=molecule_filter)

    def load_opacity(
        self,
        opacities: t.Optional[t.Union[KTable, t.List[KTable]]] = None,
        opacity_path: t.Optional[t.Union[str, t.List[str]]] = None,
        molecule_filter: t.Optional[t.Union[str, t.List[str]]] = None,
    ) -> None:
        """Main function to use when loading molecular opacities.

        Handles both cross sections and paths. Handles lists of either so lists of
        :class:`~taurex.opacity.opacity.Opacity` objects or lists of paths can be used
        to load multiple files/objects


        Parameters
        ----------
        opacities : :class:`~taurex.opacity.opacity.Opacity` or
        :obj:`list` of :class:`~taurex.opacity.opacity.Opacity` , optional
            Object(s) to include in cache

        opacity_path : str or :obj:`list` of str, optional
            search path(s) to look for molecular opacities

        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity will only be loaded
            if its molecule is in this list. Mostly used by the
            :func:`__getitem__` for filtering

        """
        from taurex.opacity import Opacity

        if opacity_path is None:
            opacity_path = self._opacity_path

        if opacities is not None:
            if isinstance(opacities, (list,)):
                self.log.debug("Opacity passed is list")
                for opacity in opacities:
                    self.add_opacity(opacity, molecule_filter=molecule_filter)
            elif isinstance(opacities, Opacity):
                self.add_opacity(opacities, molecule_filter=molecule_filter)
            else:
                self.log.error(
                    "Unknown type %s passed into opacities, should be a list, single \
                     opacity or None if reading a path",
                    type(opacities),
                )
                raise Exception("Unknown type passed into opacities")
        else:
            self.load_opacity_from_path(opacity_path, molecule_filter=molecule_filter)
            # if isinstance(opacity_path,(list,)):
            #     for path in opacity_path:
            #         self.load_opacity_from_path(path,molecule_filter=molecule_filter)

    def clear_cache(self) -> None:
        """Clears all currently loaded cross-sections."""
        self.opacity_dict = {}
        self._opacity_path = GlobalCache()["ktable_path"]
