import abc
import glob
import os
import string
from .registry import register
from .utils import inherit_docs


class MetadataExtractor(object):
    """Extracts metadata from data filenames.

    Instances of MetadataExtractor are used to extract metadata from
    filenames in bulk. Example usage:
    >>> extractor = MetadataExtractor('/path/to/data')

    Suppose the data in this directory had the following files:
    pr_*.nc, uas_*.nc, vas_*.nc
    All of the metadata lies in the data attribute:
    >>> extractor.data
    [{'filename': /path/to/data/pr_*.nc, 'variable': 'pr'},
     {'filename': /path/to/data/vas_*.nc, 'variable': 'vas'},
     {'filename': /path/to/data/uas_*.nc, 'variable': 'uas'}]

    Results can be narrowed down by specifying values for a field:
    >>> extractor.query(variable='pr')
    [{'filename': /path/to/data/pr_*.nc, 'variable': 'pr'}]

    Finally, metadata from two sets of extractors can be grouped together
    based on common field name as follows:
    >>> extractor.groupby(extractor2, 'variable')

    This class should only be used as a starting point. We recommend using
    the included `obs4MIPSExtractor` and `CORDEXExtractor`
    subclasses or creating your own subclass for your usecase.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, *paths):
        """Create a MetadataExtractor from a list of paths.

        Parameters
        ----------
        *paths
            Paths to search for data. Regex is allowed.
        """
        self.paths = paths

    @property
    def sep(self):
        return '_'

    @property
    def ext(self):
        return '.nc'

    @property
    def data(self):
        """The extracted metadata for each file, with all fields listed in the
        fields attribute included.
        """
        return self._data

    @property
    def paths(self):
        """Search paths containing the dataset files."""
        return self._paths

    @paths.setter
    def paths(self, paths):
        """Extracts the metadata from scratch when paths are reset.

        Parameters
        ----------
        paths : str or list of str
            Paths to search for data. Regex is allowed.
        """
        self._paths = paths
        self._extract()

    @property
    @abc.abstractmethod
    def fields(self):
        """The name of field in the filename, assuming the fully filtered
        filename conforms to the following convention:
        filename = <field[0]>_<field[1]>_..._<field[n]>.nc.
        Using fewer fields than the filename defines is allowed by setting the
        index attribute to select which fields to extract.
        """
        pass

    @property
    def files(self):
        """List of files (or regular expressions) for each dataset."""
        files = []
        for path in self.paths:
            if os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, f'*{self.ext}')))
            else:
                files.extend(glob.glob(path))
        return list(set(self.get_pattern(fname) for fname in files))

    @property
    def variables(self):
        """Get the list of variables included accross all the datasets."""
        return self.get_field('variable')

    @property
    def names(self):
        """Get the list of variables included accross all the datasets."""
        return self.get_field(self.name_field)

    @property
    def ignored_values(self):
        """Override this to filter out specific characters contained in a field.
        """
        return dict()

    @property
    def split_by(self):
        """Name of field which is used to split dataset into multiple files."""
        return 'time'

    @property
    def name_field(self):
        """Name of field which is used to denote dataset name."""
        return 'name'

    @property
    def index(self):
        """Indices of fieldnames in filename template.

        For example, given fields ['name', 'variable'] and index [1, 3],
        the filename template would be '{a}_{name}_{b}_{variable}.nc' In other
        words, fields a and b in the template are not extracted because they are
        not in the index list. Set to None to extract all fields in the filename.
        """
        return None

    def query(self, **kwargs):
        """Narrow down the list of files by field names.

        Parameters
        ----------
        **kwargs : dict
            key, value pairs where key is a field name.

        Returns
        -------
        data : list of dict
            Query results.

        Raises
        ------
        ValueError
            If a given field name is invalid.
        """
        fields = set(kwargs.keys())
        valid_fields = set(['filename'] + self.fields)
        if not fields.issubset(valid_fields):
            raise ValueError("Invalid fields: {}. Must be subset of: {}"
                             .format(fields, valid_fields))
        data = self.data
        for field, value in kwargs.items():
            value = value if isinstance(value, list) else [value]
            data = [meta for meta in data
                    if self._match_filter(meta, field) in value]
        return data

    def groupby(self, extractor, field):
        """Group metadata in two extractors by field name.

        Parameters
        ----------
        extractor : bcdp.extractors.MetadataExtractor
            Separate instance of MetadataExtractor to compare with.
        field : str
            Field used for grouping results.

        Returns
        -------
        groups : list of dict
            Metadata from both this instance and extractor, grouped by values of
            given field.
        """
        # First we only want to consider values of field which are contained
        # in both extractors
        subset = self.get_field(field)
        other_subset = extractor.get_field(field)
        intersection = list(subset.intersection(other_subset))

        # Next we will group the datasets in each extractor together by common
        # field values
        kwargs = {field: intersection}
        results = self.query(**kwargs)

        groups = []
        for meta in results:
            val = self._match_filter(meta, field)
            kwargs.update({field: val})
            match = extractor.query(**kwargs)
            groups.append((meta, match))

        return groups

    def get_field(self, field):
        """Returns only the selected field of the extracted data.

        Parameters
        ----------
        field : str
            Field name.

        Returns
        -------
        sub : set of str
            Set of all possible values in metadata for input field.

        Raises
        ------
        ValueError
            If a given field name is invalid.
        """
        if field not in self.fields:
            raise ValueError("Invalid field: {}. Must be one of: {}"
                             .format(field, self.fields))
        sub = set(meta[field] for meta in self.data)
        return sub

    def filter_filename(self, fname):
        """Applies a filter to each individual filename contained in the _files
        attribute.

        This is useful if some files within a dataset are known
        to not follow conventions, and "fix" them so that they do.

        Parameters
        ----------
        fname : str
            Raw input filename or regex to filter.

        Returns
        -------
        str
            Filtered filename.
        """
        return os.path.basename(fname)

    def field_index(self, field):
        """Get index of field.

        Returns
        -------
        idx : int
            Index of field.
        """
        idx = self.fields.index(field)
        if self.index:
            idx = self.index[idx]
        return idx

    def get_pattern(self, fname):
        """Group multiple file datasets together via regular expressions.

        The most common convention is to split files by time. This is the
        default value of the `split_by` attribute.

        Parameters
        ----------
        fname : str
            Raw input filename or regex to filter.

        Returns
        -------
        pattern : str
            Filtered filename.
        """
        bname = os.path.basename(fname)
        path = fname.replace(bname, '')
        base = os.path.splitext(bname)[0].split(self.sep)
        if self.split_by in self.fields:
            i = self.field_index(self.split_by)
            base[i] = '*'
        pattern = path + self.sep.join(base) + self.ext
        return pattern

    def _match_filter(self, meta, field):
        """Filter (ignore) certain character patterns when matching a field.

        Used internally.

        Parameters
        ----------
        meta : dict
            A single set of metadata from the extraction results.
        field : str
            Field name.

        Returns
        -------
        val : str
            Field value with patterns (from ignored_values attribute) removed.
        """
        val = meta[field]
        if field in self.ignored_values:
            for pattern in self.ignored_values[field]:
                val = val.replace(pattern, '')
        return val

    def _get_split_field_values(self, fname):
        """Get values associated with split_by field attribute (ie, time).

        Parameters
        ----------
        fname : str
            Dataset filename with split_field replaced with regex (eg '*')

        Returns
        -------
        values : list
            List of values of split-by field in expanded fname.
        """
        files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(fname)]
        filt = [self.filter_filename(f) for f in files]
        values = [f.split(self.sep)[self.field_index(self.split_by)] for f in filt]
        return sorted(values)

    def _extract(self):
        """Do the actual metadata extraction.

        filenames can also be filtered via filter_filename() to remove unwanted
        characters from the extraction.
        """
        self._data = []
        for fname in self.files:
            meta = dict(filename=fname)

            # Perform the actual metadata extraction
            fname = os.path.splitext(self.filter_filename(fname))[0]
            values = fname.split(self.sep)

            # Handle the case where number of fields is less than the length
            # of the extracted values, ie cases where we only want to extract
            # a subset of available fields.
            if self.index:
                values = [val for i, val in enumerate(values) if i in self.index]

            meta.update(dict(zip(self.fields, values)))
            if self.split_by in self.fields:
                meta[self.split_by] = self._get_split_field_values(meta['filename'])
            self._data.append(meta)


@register('metadata.obs4MIPs')
@inherit_docs
class obs4MIPSExtractor(MetadataExtractor):
    @property
    def name_field(self):
        return 'instrument'

    @property
    def instruments(self):
        """Get the list of instruments accross all the datasets."""
        return self.get_field('instrument')

    @property
    def fields(self):
        """obs4MIPs fields"""
        fields = ['variable', 'instrument', 'processing_level', 'version',
                  'time']
        return fields

    @property
    def ignored_values(self):
        return dict(variable=['calipso', 'Lidarsr532'])

    def filter_filename(self, fname):
        # Overriden to deal with some filenames that don't follow regular
        # conventions (TRMM, CALIPSO, AVISO)
        if '_obs4MIPS_' in fname:
            # Time in CALIPSO files are separated by _ rather than -
            fname = os.path.splitext(fname)[0]
            parts = fname.split(self.sep)
            time = '-'.join(parts[-2:-1])
            fname = self.sep.join([*parts[:-2], time, self.ext])
    
        if 'AVISO' in fname:
            fname = os.path.splitext(fname)[0]
            parts = fname.split(self.sep)
            fname = self.sep.join([*parts[:3], 'v1', parts[-1]])
    
        fname = (os.path
                   .basename(fname)
                   .replace('_obs4MIPs_', self.sep)
                   .replace('TRMM-L3', 'TRMM_L3'))
        return fname

    def get_pattern(self, fname):
        # Overriden to account for CALIPSO filenames having extra metadata
        base = fname.split(self.sep)
        offset = -2 if len(base) != 5 else -1
        pattern = self.sep.join(base[:offset] + ['*' + self.ext])
        return pattern


@register('metadata.CORDEX')
class CORDEXExtractor(MetadataExtractor):
    @property
    def name_field(self):
        return 'model'

    @property
    def models(self):
        """Get the list of models accross all the datasets."""
        return self.get_field('model')

    @property
    def fields(self):
        """CORDEX fields"""
        fields = ['variable', 'domain', 'driving_model', 'experiment',
                  'ensemble', 'model', 'version', 'time_step', 'time']
        return fields


@register('metadata.CMIP5')
class CMIP5Extractor(MetadataExtractor):
    @property
    def name_field(self):
        return 'model'

    @property
    def models(self):
        """Get the list of models accross all the datasets."""
        return self.get_field('model')

    @property
    def fields(self):
        """CMIP5 fields"""
        fields = ['variable', 'temporal_resolution', 'model', 'experiment',
                  'ensemble', 'time']
        return fields

def build_extractor(name, template, split_by=None, name_field=None, index=None,
                    filter_filename=None, **ignored_values):
    """Build a metadata extractor subclass from a template string.

    Primarily intended to be a useful shortcut for implementing custom subclasses
    of MetaDataExtractor, in which the required attributes are inferred via a
    template string.

    Parameters
    ----------
    name : str
        Name of the extractor subclass. This will also be used as the registry
        key, so that the class may be accessed from the registry via
        registry['metadata'][name]. At the bare minimum, a dataset name and
        variable field are required in the template.
    template : str
        File-name conventions template, in the form:
        `"{<field_1>}<sep>{<field_2>}<sep>...{<field_n>}.<ext>"`, where `sep` is
        the field separator (delimiter, usually '_') and <ext> is the file
        extension, typically 'nc'. For convenience, templates can be shortened
        in regex style as well (eg, `"{*_{<field_2>}_*{<field_5>}.nc"`), provided
        `index=[1, 4]` is passed to this function.
    split_by : str, optional
        Name of the field in the template used to split multiple files. Note
        that the split_by field need not be in the template if no splitting is
        required. Default is `'time'`.
    name_field : str, optional
        Field which denotes the name of the dataset (eg `'name'`, `'model'`).
    index : list of int, optional
        Indices of fieldnames to extract in filename.
    filter_filename : callable
        Function which takesa filename and returns a modified filename from which
        the metadata is extracted from.
    **ignored_values : dict, optional
        Key, value pairs where keys are field names and values are patterns
        to filter out from given values of that field. Particularly useful when
        some files follow slightly different conventions from the rest.

    Returns
    -------
    class(MetaDataExtractor)
        MetaDataExtractor subclass.
    """
    formatter = string.Formatter()
    fields = [v[1] for v in formatter.parse(template) if v[1]]
    for field in fields:
        template = template.replace(field, '')
    template = template.replace('{}', ' ').replace('*', ' ')
    props = template.split()
    sep = props[0]
    ext = props[-1] if props[-1].startswith('.') else None
    return build_extractor_from_params(name, fields, sep=sep, ext=ext,
                                       split_by=split_by, name_field=name_field,
                                       index=index, filter_filename=filter_filename,
                                       **ignored_values)


def build_extractor_from_params(name, fields, sep=None, ext=None, split_by=None,
                                name_field=None, index=None, filter_filename=None,
                                **ignored_values):
    """Build a metadata extractor subclass from a template string.

    Primarily intended to be a useful shortcut for implementing custom subclasses
    of MetaDataExtractor, in which the required attributes are inferred via a
    template string.

    Parameters
    ----------
    name : str
        Name of the extractor subclass. This will also be used as the registry
        key, so that the class may be accessed from the registry via
        `registry['metadata'][name]`.
    fields : list of str
        List of field names to extract, in order from left to right of the filename.
        At the bare minimum, a dataset name and variable field are required in
        the template.
    sep : str, optional
        Field separator in the filename (delimiter, Default: `'_'`).
    ext : str, optional
        File extension (Default: `'nc'`)
    split_by : str, optional
        Name of the field in the template used to split multiple files. Note
        that the split_by field need not be in the template if no splitting is
        required. Default is `'time'`.
    name_field : str, optional
        Field which denotes the name of the dataset (eg `'model'`).
        Default is the very first field.
    index : list of int, optional
        Indices of fieldnames to extract in filename.
    **ignored_values : dict, optional
        Key, value pairs where keys are field names and values are patterns
        to filter out from given values of that field. Particularly useful when
        some files follow slightly different conventions from the rest.

    Returns
    -------
    class(MetaDataExtractor)
        MetaDataExtractor subclass.
    """
    sep = sep if sep else '_'
    ext = ext if ext else '.nc'
    split_by = split_by if split_by else 'time'
    name_field = name_field if name_field else fields[0]
    @register(f'metadata.{name}')
    class GenericExtractor(MetadataExtractor):
        @property
        def fields(self):
            return fields

        @property
        def sep(self):
            return sep

        @property
        def ext(self):
            return ext

        @property
        def split_by(self):
            return split_by

        @property
        def name_field(self):
            return name_field

        @property
        def ignored_values(self):
            return ignored_values

        @property
        def index(self):
            return index
            
        def filter_filename(self, fname):
            fname = os.path.basename(fname)
            if filter_filename:
                return filter_filename(fname)
            return fname

    GenericExtractor.__name__ = name
    return GenericExtractor
