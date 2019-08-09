import collections
import functools

import pyspark.sql.functions as F
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.window import Window

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as types
from ibis.pyspark.operations import PysparkTable
from ibis.sql.compiler import Dialect

_operation_registry = {}


class PysparkExprTranslator:
    _registry = _operation_registry

    @classmethod
    def compiles(cls, klass):
        def decorator(f):
            cls._registry[klass] = f
            return f

        return decorator

    def translate(self, expr, **kwargs):
        # The operation node type the typed expression wraps
        op = expr.op()

        if type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, expr, **kwargs)
        else:
            raise com.OperationNotDefinedError(
                'No translation rule for {}'.format(type(op))
            )


class PysparkDialect(Dialect):
    translator = PysparkExprTranslator


compiles = PysparkExprTranslator.compiles


@compiles(PysparkTable)
def compile_datasource(t, expr):
    op = expr.op()
    name, _, client = op.args
    return client._session.table(name)


@compiles(ops.Selection)
def compile_selection(t, expr):
    op = expr.op()

    src_table = t.translate(op.table)
    col_names_in_selection_order = []
    for selection in op.selections:
        if isinstance(selection, types.TableExpr):
            col_names_in_selection_order.extend(selection.columns)
        elif isinstance(selection, types.ColumnExpr):
            column_name = selection.get_name()
            col_names_in_selection_order.append(column_name)
            if column_name not in src_table.columns:
                column = t.translate(selection)
                src_table = src_table.withColumn(column_name, column)

    return src_table[col_names_in_selection_order]


@compiles(ops.TableColumn)
def compile_column(t, expr):
    op = expr.op()
    return t.translate(op.table)[op.name]


@compiles(ops.DistinctColumn)
def compile_distinct(t, expr):
    op = expr.op()
    src_table = t.translate(op.arg.to_projection())
    src_column_name = op.arg.get_name()
    return src_table.select(src_column_name).distinct()[src_column_name]


@compiles(ops.SelfReference)
def compile_self_reference(t, expr):
    op = expr.op()
    return t.translate(op.table)


@compiles(ops.Equals)
def compile_equals(t, expr):
    op = expr.op()
    return t.translate(op.left) == t.translate(op.right)


@compiles(ops.Greater)
def compile_greater(t, expr):
    op = expr.op()
    return t.translate(op.left) > t.translate(op.right)


@compiles(ops.GreaterEqual)
def compile_greater_equal(t, expr):
    op = expr.op()
    return t.translate(op.left) >= t.translate(op.right)


@compiles(ops.Multiply)
def compile_multiply(t, expr):
    op = expr.op()
    return t.translate(op.left) * t.translate(op.right)


@compiles(ops.Subtract)
def compile_subtract(t, expr):
    op = expr.op()
    return t.translate(op.left) - t.translate(op.right)


@compiles(ops.Literal)
def compile_literal(t, expr):
    value = expr.op().value

    if isinstance(value, collections.abc.Set):
        # Don't wrap set with F.lit
        if isinstance(value, frozenset):
            # Spark doens't like frozenset
            return set(value)
        else:
            return value
    else:
        return F.lit(expr.op().value)


@compiles(ops.Aggregation)
def compile_aggregation(t, expr):
    op = expr.op()

    src_table = t.translate(op.table)
    aggs = [t.translate(m, context="agg")
            for m in op.metrics]

    if op.by:
        bys = [t.translate(b) for b in op.by]
        return src_table.groupby(*bys).agg(*aggs)
    else:
        return src_table.agg(*aggs)


@compiles(ops.Contains)
def compile_contains(t, expr):
    col = t.translate(expr.op().value)
    return col.isin(t.translate(expr.op().options))


def compile_aggregator(t, expr, fn, context=None):
    op = expr.op()
    src_col = t.translate(op.arg)

    if getattr(op, "where", None) is not None:
        condition = t.translate(op.where)
        src_col = F.when(condition, src_col)

    col = fn(src_col)
    if context:
        return col
    else:
        return t.translate(expr.op().arg.op().table).select(col)


@compiles(ops.GroupConcat)
def compile_group_concat(t, expr, context=None):
    sep = expr.op().sep.op().value

    def fn(col):
        return F.concat_ws(sep, F.collect_list(col))
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Any)
def compile_any(t, expr, context=None):
    return compile_aggregator(t, expr, F.max, context)


@compiles(ops.NotAny)
def compile_notany(t, expr, context=None):

    def fn(col):
        return ~F.max(col)
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.All)
def compile_all(t, expr, context=None):
    return compile_aggregator(t, expr, F.min, context)


@compiles(ops.NotAll)
def compile_notall(t, expr, context=None):

    def fn(col):
        return ~F.min(col)
    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Count)
def compile_count(t, expr, context=None):
    return compile_aggregator(t, expr, F.count, context)


@compiles(ops.Max)
def compile_max(t, expr, context=None):
    return compile_aggregator(t, expr, F.max, context)


@compiles(ops.Min)
def compile_min(t, expr, context=None):
    return compile_aggregator(t, expr, F.min, context)


@compiles(ops.Mean)
def compile_mean(t, expr, context=None):
    return compile_aggregator(t, expr, F.mean, context)


@compiles(ops.Sum)
def compile_sum(t, expr, context=None):
    return compile_aggregator(t, expr, F.sum, context)


@compiles(ops.StandardDev)
def compile_std(t, expr, context=None):
    how = expr.op().how

    if how == 'sample':
        fn = F.stddev_samp
    elif how == 'pop':
        fn = F.stddev_pop
    else:
        raise AssertionError("Unexpected how: {}".format(how))

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Variance)
def compile_variance(t, expr, context=None):
    how = expr.op().how

    if how == 'sample':
        fn = F.var_samp
    elif how == 'pop':
        fn = F.var_pop
    else:
        raise AssertionError("Unexpected how: {}".format(how))

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.Arbitrary)
def compile_arbitrary(t, expr, context=None):
    how = expr.op().how

    if how == 'first':
        fn = functools.partial(F.first, ignorenulls=True)
    elif how == 'last':
        fn = functools.partial(F.last, ignorenulls=True)
    else:
        raise NotImplementedError

    return compile_aggregator(t, expr, fn, context)


@compiles(ops.WindowOp)
def compile_window_op(t, expr):
    op = expr.op()
    return t.translate(op.expr).over(compile_window(op.window))


@compiles(ops.Greatest)
def compile_greatest(t, expr):
    op = expr.op()

    src_columns = t.translate(op.arg)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.greatest(*src_columns)


@compiles(ops.Least)
def compile_least(t, expr):
    op = expr.op()

    src_columns = t.translate(op.arg)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.least(*src_columns)


@compiles(ops.Abs)
def compile_abs(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.abs(src_column)


@compiles(ops.Round)
def compile_round(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    scale = op.digits.op().value if op.digits is not None else 0
    rounded = F.round(src_column, scale=scale)
    if scale == 0:
        rounded = rounded.astype('long')
    return rounded


@compiles(ops.Ceil)
def compile_ceil(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.ceil(src_column)


@compiles(ops.Floor)
def compile_floor(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.floor(src_column)


@compiles(ops.Exp)
def compile_exp(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.exp(src_column)


@compiles(ops.Sign)
def compile_sign(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)

    return F.when(src_column == 0, F.lit(0.0)) \
        .otherwise(F.when(src_column > 0, F.lit(1.0)).otherwise(-1.0))


@compiles(ops.Sqrt)
def compile_sqrt(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.sqrt(src_column)


@compiles(ops.Log)
def compile_log(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.log(float(op.base.op().value), src_column)


@compiles(ops.Ln)
def compile_ln(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.log(src_column)


@compiles(ops.Log2)
def compile_log2(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.log2(src_column)


@compiles(ops.Log10)
def compile_log10(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.log10(src_column)


@compiles(ops.Modulus)
def compile_modulus(t, expr):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return left % right


@compiles(ops.Negate)
def compile_negate(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return -src_column


@compiles(ops.Add)
def compile_add(t, expr):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return left + right


@compiles(ops.Divide)
def compile_divide(t, expr):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return left / right


@compiles(ops.FloorDivide)
def compile_floor_divide(t, expr):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return F.floor(left / right)


@compiles(ops.Power)
def compile_power(t, expr):
    op = expr.op()

    left = t.translate(op.left)
    right = t.translate(op.right)
    return F.pow(left, right)


@compiles(ops.IsNan)
def compile_isnan(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.isnan(src_column)


@compiles(ops.IsInf)
def compile_isinf(t, expr):
    import numpy as np
    op = expr.op()

    @pandas_udf('boolean', PandasUDFType.SCALAR)
    def isinf(v):
        return np.isinf(v)

    src_column = t.translate(op.arg)
    return isinf(src_column)


@compiles(ops.Uppercase)
def compile_uppercase(t, expr):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def upper(v):
        return v.str.upper()

    src_column = t.translate(op.arg)
    return upper(src_column)


@compiles(ops.Lowercase)
def compile_lowercase(t, expr):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def lower(v):
        return v.str.lower()

    src_column = t.translate(op.arg)
    return lower(src_column)


@compiles(ops.Reverse)
def compile_reverse(t, expr):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def reverse(s):
        return s.str[::-1]

    src_column = t.translate(op.arg)
    return reverse(src_column)


@compiles(ops.Strip)
def compile_strip(t, expr):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def strip(s):
        return s.str.strip()

    src_column = t.translate(op.arg)
    return strip(src_column)


@compiles(ops.LStrip)
def compile_lstrip(t, expr):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def lstrip(s):
        return s.str.lstrip()

    src_column = t.translate(op.arg)
    return lstrip(src_column)


@compiles(ops.RStrip)
def compile_rstrip(t, expr):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def rstrip(s):
        return s.str.lstrip()

    src_column = t.translate(op.arg)
    return rstrip(src_column)


@compiles(ops.Capitalize)
def compile_capitalize(t, expr):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def capitalize(s):
        return s.str.capitalize()

    src_column = t.translate(op.arg)
    return capitalize(src_column)


@compiles(ops.Substring)
def compile_substring(t, expr):
    op = expr.op()

    @F.udf('string')
    def substring(s, start, length):
        end = start + length
        return s[start:end]

    src_column = t.translate(op.arg)
    start_column = t.translate(op.start)
    length_column = t.translate(op.length)
    return substring(src_column, start_column, length_column)


@compiles(ops.StringLength)
def compile_string_length(t, expr):
    op = expr.op()

    @pandas_udf('int', PandasUDFType.SCALAR)
    def length(s):
        return s.str.len()

    src_column = t.translate(op.arg)
    return length(src_column)


@compiles(ops.StrRight)
def compile_str_right(t, expr):
    op = expr.op()

    @F.udf('string')
    def str_right(s, nchars):
        return s[-nchars:]

    src_column = t.translate(op.arg)
    nchars_column = t.translate(op.nchars)
    return str_right(src_column, nchars_column)


@compiles(ops.Repeat)
def compile_repeat(t, expr):
    op = expr.op()

    @F.udf('string')
    def repeat(s, times):
        return s * times

    src_column = t.translate(op.arg)
    times_column = t.translate(op.times)
    return repeat(src_column, times_column)


@compiles(ops.StringFind)
def compile_string_find(t, expr):
    op = expr.op()

    @F.udf('long')
    def str_find(s, substr, start, end):
        return s.find(substr, start, end)

    src_column = t.translate(op.arg)
    substr_column = t.translate(op.substr)
    start_column = t.translate(op.start) if op.start else F.lit(None)
    end_column = t.translate(op.end) if op.end else F.lit(None)
    return str_find(src_column, substr_column, start_column, end_column)


@compiles(ops.Translate)
def compile_translate(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    from_str = op.from_str.op().value
    to_str = op.to_str.op().value
    return F.translate(src_column, from_str, to_str)


@compiles(ops.LPad)
def compile_lpad(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.lpad(src_column, length, pad)


@compiles(ops.RPad)
def compile_rpad(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.rpad(src_column, length, pad)


@compiles(ops.StringJoin)
def compile_string_join(t, expr):
    op = expr.op()

    @F.udf('string')
    def join(sep, arr):
        return sep.join(arr)

    sep_column = t.translate(op.sep)
    arg = t.translate(op.arg)
    return join(sep_column, F.array(arg))


@compiles(ops.RegexSearch)
def compile_regex_search(t, expr):
    import re
    op = expr.op()

    @F.udf('boolean')
    def regex_search(s, pattern):
        return True if re.search(pattern, s) else False

    src_column = t.translate(op.arg)
    pattern = t.translate(op.pattern)
    return regex_search(src_column, pattern)


@compiles(ops.RegexExtract)
def compile_regex_extract(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    pattern = op.pattern.op().value
    idx = op.index.op().value
    return F.regexp_extract(src_column, pattern, idx)


@compiles(ops.RegexReplace)
def compile_regex_replace(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    pattern = op.pattern.op().value
    replacement = op.replacement.op().value
    return F.regexp_replace(src_column, pattern, replacement)


@compiles(ops.StringReplace)
def compile_string_replace(t, expr):
    return compile_regex_replace(t, expr)


@compiles(ops.StringSplit)
def compile_string_split(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    delimiter = op.delimiter.op().value
    return F.split(src_column, delimiter)


@compiles(ops.StringAscii)
def compile_string_ascii(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    return F.ascii(src_column)


@compiles(ops.StringSQLLike)
def compile_string_like(t, expr):
    op = expr.op()

    src_column = t.translate(op.arg)
    pattern = op.pattern.op().value
    return src_column.like(pattern)


@compiles(ops.ValueList)
def compile_value_list(t, expr):
    op = expr.op()
    return [t.translate(col) for col in op.values]


@compiles(ops.InnerJoin)
def compile_inner_join(t, expr):
    return compile_join(t, expr, 'inner')


def compile_join(t, expr, how):
    op = expr.op()

    left_df = t.translate(op.left)
    right_df = t.translate(op.right)
    # TODO: Handle multiple predicates
    predicates = t.translate(op.predicates[0])

    return left_df.join(right_df, predicates, how)


# Cannot register with @compiles because window doesn't have an
# op() object
def compile_window(expr):
    spark_window = Window.partitionBy()
    return spark_window


t = PysparkExprTranslator()


def translate(expr):
    return t.translate(expr)
