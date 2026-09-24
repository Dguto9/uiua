use std::slice::{Iter, IterMut};

use super::*;

#[derive(Debug, Clone, PartialOrd, Serialize, Deserialize)]
pub enum TypeVal {
    Num(f64),
    NumList(EcoVec<f64>),
    Val(Value),
    Type(Type),
}

impl TypeVal {
    pub fn ty(self) -> Type {
        match self {
            TypeVal::Num(_) => Scalar::Num.scalar_type(),
            TypeVal::NumList(list) => Scalar::Num.shaped(list.len()),
            TypeVal::Type(ty) => ty.clone(),
            TypeVal::Val(val) => Type::of_val(&val),
        }
    }
    pub fn spec(self) -> Option<Type> {
        match self {
            TypeVal::Num(n) => Some(Scalar::Any.shaped(num_as_dim(n)?)),
            TypeVal::NumList(list) => Some(Scalar::Any.shaped(Option::<DynShape>::from_iter(
                list.into_iter().map(num_as_dim),
            )?)),
            TypeVal::Val(val) => Type::from_spec(&val),
            tv => Some(tv.ty()),
        }
    }
    pub fn row_count(&self) -> Dim {
        match self {
            TypeVal::Num(_) => Dim::Static(1),
            TypeVal::NumList(list) => Dim::Static(list.len()),
            TypeVal::Val(value) => Dim::Static(value.row_count()),
            TypeVal::Type(ty) => ty.shape.row_count(),
        }
    }
    pub fn shape(&self) -> Cow<DynShape> {
        match self {
            TypeVal::Num(_) => Cow::Owned(DynShape::SCALAR),
            TypeVal::NumList(items) => Cow::Owned([items.len()].into()),
            TypeVal::Val(value) => Cow::Owned((&value.shape).into()),
            TypeVal::Type(ty) => Cow::Borrowed(&ty.shape),
        }
    }
    pub fn scalar(&self) -> Scalar {
        match self {
            TypeVal::Num(n) => slice::from_ref(n).into(),
            TypeVal::NumList(nums) => nums.as_slice().into(),
            TypeVal::Val(val) => Scalar::of_val(val),
            TypeVal::Type(ty) => ty.scalar.clone(),
        }
    }
    pub fn set_scalar(&mut self, scalar: Scalar) {
        if scalar.superset_of(&self.scalar()) {
            return;
        }
        match (self, scalar) {
            (TypeVal::Num(_) | TypeVal::NumList(_), Scalar::Num) => {}
            (TypeVal::Val(val), scalar) if Type::of_val(&*val).scalar == scalar => {}
            (tv, scalar) => {
                let mut ty = take(tv).ty();
                ty.scalar = scalar;
                *tv = ty.into();
            }
        }
    }
    pub fn set_shape(&mut self, shape: DynShape) {
        let curr = self.shape();
        if *curr == shape {
            return;
        }
        let mut ty = take(self).ty();
        ty.shape = shape;
        *self = ty.into();
    }
    pub fn rank(&self) -> usize {
        match self {
            TypeVal::Num(_) => 0,
            TypeVal::NumList(_) => 1,
            TypeVal::Val(val) => val.rank(),
            TypeVal::Type(ty) => ty.shape.rank(),
        }
    }
    pub fn leading_rank(&self) -> usize {
        match self {
            TypeVal::Num(_) => 0,
            TypeVal::NumList(_) => 1,
            TypeVal::Val(val) => val.rank(),
            TypeVal::Type(ty) => ty.shape.dims.len(),
        }
    }
    pub fn suffix_rank(&self) -> Option<usize> {
        match self {
            TypeVal::Type(ty) => ty.shape.suffix.as_ref().map(|suf| suf.len()),
            _ => None,
        }
    }
    pub fn into_row(self) -> Self {
        match self {
            TypeVal::Num(_) => self,
            TypeVal::NumList(list) if list.len() == 1 => TypeVal::Num(list[0]),
            TypeVal::Val(mut val) if val.row_count() == 1 => {
                val.shape.undo_fix();
                TypeVal::Val(val)
            }
            TypeVal::NumList(_) | TypeVal::Val(_) => TypeVal::Type(self.ty().into_row()),
            TypeVal::Type(ty) => TypeVal::Type(ty.into_row()),
        }
    }
    pub fn into_first_row(self) -> Self {
        match self {
            TypeVal::NumList(list) if !list.is_empty() => TypeVal::Num(list[0]),
            TypeVal::Type(ty) => TypeVal::Type(ty.into_first_row()),
            TypeVal::Val(val) if val.row_count() > 0 => {
                TypeVal::Val(val.into_rows().next().unwrap())
            }
            tv => tv.into_row(),
        }
    }
    pub fn into_last_row(self) -> Self {
        match self {
            TypeVal::NumList(list) if !list.is_empty() => TypeVal::Num(*list.last().unwrap()),
            TypeVal::Type(ty) => TypeVal::Type(ty.into_last_row()),
            TypeVal::Val(val) if val.row_count() > 0 => {
                TypeVal::Val(val.into_rows().next_back().unwrap())
            }
            tv => tv.into_row(),
        }
    }
    pub fn prepend_dim(&mut self, dim: Dim) {
        match (&mut *self, dim) {
            (TypeVal::Num(n), Dim::Static(1)) => *self = TypeVal::NumList(eco_vec![*n]),
            (TypeVal::NumList(list), Dim::Static(1)) => {
                let mut val: Value = take(list).into();
                val.fix();
                *self = TypeVal::Val(val)
            }
            (TypeVal::Val(val), Dim::Static(1)) => val.fix(),
            (TypeVal::Type(ty), _) => ty.shape.dims.insert(0, dim),
            (_, Dim::Dyn | Dim::Static(_)) => {
                *self = take(self).ty().into();
                self.prepend_dim(dim);
            }
        }
    }
    pub fn boxed(self) -> Self {
        match self {
            TypeVal::Num(n) => TypeVal::Val(Value::from(n).boxed_if(true)),
            TypeVal::NumList(list) => TypeVal::Val(Value::from(list).boxed_if(true)),
            TypeVal::Val(val) => TypeVal::Val(val.boxed_if(true)),
            tv => Scalar::Box(ScalarBox::All(Box::new(OrType::from(tv.ty()))))
                .scalar_type()
                .into(),
        }
    }
    pub fn unboxed(self) -> Self {
        match self {
            TypeVal::Val(val) => TypeVal::Val(val.unboxed()),
            // TypeVal::Type(Type {
            //     scalar: Scalar::Box(sb),
            //     shape,
            // }) if shape.is_scalar() => sb.into_inner().map_or_else(TypeVal::default, Into::into),
            tv => tv,
        }
    }
    pub fn is_boxed(&self) -> bool {
        match self {
            TypeVal::Type(ty) => matches!(ty.scalar, Scalar::Box(_)) && ty.shape.is_scalar(),
            TypeVal::Val(Value::Box(arr)) => arr.rank() == 0,
            _ => false,
        }
    }
    pub fn as_boxes(&self) -> Option<(Option<EcoString>, Vec<Self>)> {
        match self {
            TypeVal::Type(ty) => ty
                .as_boxes()
                .map(|(name, v)| (name, v.into_iter().map(Into::into).collect())),
            TypeVal::Val(Value::Box(arr)) => Some((
                arr.meta.label.as_ref().map(Into::into),
                arr.data.iter().map(|Boxed(v)| v.clone().into()).collect(),
            )),
            _ => None,
        }
    }
    pub fn as_scalar_spec(&self) -> Option<Scalar> {
        match self {
            TypeVal::Val(val) => value_as_scalar_spec(val),
            _ => None,
        }
    }
    pub fn as_dim(&self) -> Option<Dim> {
        let n = match self {
            TypeVal::Num(n) => *n,
            TypeVal::Val(Value::Num(arr)) if arr.rank() == 0 => arr.data[0],
            TypeVal::Val(Value::Byte(arr)) if arr.rank() == 0 => arr.data[0] as f64,
            _ => return None,
        };
        if n.is_infinite() {
            Some(Dim::Dyn)
        } else if n.fract() == 0.0 && n >= 0.0 {
            Some(Dim::Static(n as usize))
        } else {
            None
        }
    }
    pub fn as_dims(&self) -> Option<Vec<Dim>> {
        Some(match self {
            TypeVal::NumList(nums)
                if (nums.iter()).all(|&n| n == f64::INFINITY || n >= 0.0 && n.fract() == 0.0) =>
            {
                nums.iter().copied().map(Into::into).collect()
            }
            TypeVal::Val(Value::Num(arr))
                if arr.rank() == 1
                    && (arr.data.iter())
                        .all(|&n| n == f64::INFINITY || n >= 0.0 && n.fract() == 0.0) =>
            {
                arr.data.iter().copied().map(Into::into).collect()
            }
            TypeVal::Val(Value::Byte(arr)) if arr.rank() == 1 => {
                arr.data.iter().map(|&f| Dim::Static(f as usize)).collect()
            }
            _ => return None,
        })
    }
    pub fn is_any(&self) -> bool {
        matches!(self, TypeVal::Type(ty) if ty.is_any())
    }
    pub fn reshape_scalar(&mut self, dim: Dim, suffix: bool) {
        match (dim, suffix) {
            (Dim::Static(n), false) => match self {
                TypeVal::Num(x) => *self = eco_vec![*x; n].into(),
                TypeVal::NumList(list) => {
                    let mut val: Value = Value::from(take(list));
                    val.reshape_scalar(Ok(n as isize), false, Context::NONE)
                        .unwrap();
                    *self = val.into()
                }
                TypeVal::Val(val) => val
                    .reshape_scalar(Ok(n as isize), false, Context::NONE)
                    .unwrap(),
                TypeVal::Type(ty) => ty.shape.dims.insert(0, dim),
            },
            _ => {
                let mut ty = take(self).ty();
                if suffix {
                    ty.shape.suffix.get_or_insert_default()
                } else {
                    &mut ty.shape.dims
                }
                .insert(0, dim);
                *self = ty.into();
            }
        }
    }
    // // Deduplicate and remove nesting
    // fn create_or(variants: impl IntoIterator<Item = TypeVal>) -> TypeVal {
    //     TypeVal::Or(
    //         variants
    //             .into_iter()
    //             .flat_map(|v| match v {
    //                 TypeVal::Or(v) => v.iter().into(),
    //                 v => std::iter::once(v).into(),
    //             })
    //             .collect::<HashSet<_>>()
    //             .into_iter()
    //             .collect::<Vec<_>>(),
    //     )
    // }
    fn superset_of(&self, other: &Self) -> bool {
        match (self, other) {
            (TypeVal::Num(a), TypeVal::Num(b)) => a == b,
            (TypeVal::NumList(a), TypeVal::NumList(b)) => a == b,
            (TypeVal::Val(a), TypeVal::Val(b)) => a == b,
            (TypeVal::Type(ty), TypeVal::Num(_)) => {
                matches!(ty.scalar, Scalar::Num) && ty.shape.is_scalar()
            }
            (TypeVal::Type(ty), TypeVal::NumList(_)) => {
                matches!(ty.scalar, Scalar::Num) && ty.shape.rank() == 1
            }
            (TypeVal::Type(ty), TypeVal::Val(val)) => ty.superset_of(&Type::of_val(val)),
            (TypeVal::Type(ty1), TypeVal::Type(ty2)) => ty1.superset_of(ty2),
            _ => false,
        }
    }
}

impl Default for TypeVal {
    fn default() -> Self {
        TypeVal::Type(Type::default())
    }
}

impl PartialEq for TypeVal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Num(a), Self::Num(b)) => a.array_eq(b),
            (Self::NumList(a), Self::NumList(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.array_eq(b))
            }
            (Self::Val(a), Self::Val(b)) => a == b,
            (Self::Type(a), Self::Type(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for TypeVal {}

impl From<Value> for TypeVal {
    fn from(value: Value) -> Self {
        match value {
            Value::Num(arr) if arr.rank() == 0 => TypeVal::Num(arr.data[0]),
            Value::Byte(arr) if arr.rank() == 0 => TypeVal::Num(arr.data[0] as f64),
            Value::Num(arr) if arr.rank() == 1 => TypeVal::NumList(arr.data.into()),
            Value::Byte(arr) if arr.rank() == 1 => {
                TypeVal::NumList(arr.data.into_iter().map(Into::into).collect())
            }
            val => TypeVal::Val(val),
        }
    }
}

impl From<Type> for TypeVal {
    fn from(ty: Type) -> Self {
        TypeVal::Type(ty)
    }
}

impl From<Scalar> for TypeVal {
    fn from(scalar: Scalar) -> Self {
        TypeVal::Type(scalar.into())
    }
}

impl From<usize> for TypeVal {
    fn from(n: usize) -> Self {
        (n as f64).into()
    }
}

impl From<u8> for TypeVal {
    fn from(n: u8) -> Self {
        (n as f64).into()
    }
}

impl From<f64> for TypeVal {
    fn from(n: f64) -> Self {
        TypeVal::Num(n)
    }
}

impl From<EcoVec<f64>> for TypeVal {
    fn from(list: EcoVec<f64>) -> Self {
        TypeVal::NumList(list)
    }
}

impl From<Complex> for TypeVal {
    fn from(_: Complex) -> Self {
        Scalar::Complex.into()
    }
}

impl From<Dim> for TypeVal {
    fn from(dim: Dim) -> Self {
        match dim {
            Dim::Static(n) => n.into(),
            Dim::Dyn => Scalar::Nat.into(),
        }
    }
}

impl From<DynShape> for TypeVal {
    fn from(shape: DynShape) -> Self {
        if shape.suffix.is_some() {
            TypeVal::Type(Scalar::Nat.shaped(Dim::Dyn))
        } else {
            let mut list = EcoVec::new();
            for dim in shape.dims {
                match dim {
                    Dim::Static(n) => list.push(n as f64),
                    Dim::Dyn => return TypeVal::Type(Scalar::Nat.shaped(Dim::Dyn)),
                }
            }
            TypeVal::NumList(list)
        }
    }
}

impl fmt::Display for TypeVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeVal::Num(n) => write!(f, "{}", n.grid_string(false)),
            TypeVal::NumList(list) => {
                write!(f, "[")?;
                for (i, n) in list.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", n.grid_string(false))?;
                }
                write!(f, "]")
            }
            TypeVal::Val(val) => {
                let s = val.grid_string(false);
                if s.contains("\n") || s.len() > 10 {
                    Type::of_val(val).fmt(f)
                } else {
                    write!(f, "{s}")
                }
            }
            TypeVal::Type(ty) => ty.fmt(f),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Serialize, Deserialize)]
pub struct OrTypeVal(Vec<TypeVal>);
impl From<TypeVal> for OrTypeVal {
    fn from(value: TypeVal) -> Self {
        OrTypeVal(vec![value])
    }
}

impl From<Scalar> for OrTypeVal {
    fn from(value: Scalar) -> Self {
        TypeVal::Type(value.scalar_type()).into()
    }
}

impl From<OrType> for OrTypeVal {
    fn from(value: OrType) -> Self {
        value.into_iter().map(TypeVal::from).collect()
    }
}

impl FromIterator<TypeVal> for OrTypeVal {
    fn from_iter<T: IntoIterator<Item = TypeVal>>(iter: T) -> Self {
        let types = iter.into_iter().collect::<Vec<_>>();
        OrTypeVal(
            types
                .iter()
                .enumerate()
                .flat_map(|(i, t1)| {
                    types
                        .iter()
                        .cloned()
                        .enumerate()
                        .filter(move |(j, t2)| i != *j && t1.superset_of(t2))
                })
                .map(|i| i.1)
                .collect(),
        )
    }
}

impl IntoIterator for OrTypeVal {
    type Item = TypeVal;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Default for OrTypeVal {
    fn default() -> Self {
        vec![TypeVal::default()].into_iter().collect()
    }
}

impl OrTypeVal {
    pub fn iter_mut(&mut self) -> IterMut<TypeVal> {
        self.0.iter_mut()
    }
    pub fn iter(&self) -> Iter<TypeVal> {
        self.0.iter()
    }
    pub fn branch_count(&self) -> usize {
        self.0.len()
    }
    pub fn ty(self) -> OrType {
        self.into_iter().map(TypeVal::ty).collect()
    }
    pub fn spec(self) -> Option<OrType> {
        self.into_iter().map(TypeVal::spec).collect()
    }
    pub fn row_count(&self) -> Dim {
        match self.0.as_slice() {
            [only] => only.row_count(),
            _ => Dim::Dyn,
        }
    }
    pub fn shape(&self) -> Cow<DynShape> {
        match self.0.as_slice() {
            [only] => only.shape(),
            _ => Cow::Owned(DynShape::ANY),
        }
    }
    pub fn set_scalar(&mut self, scalar: Scalar) {
        self.iter_mut().for_each(|tv| tv.set_scalar(scalar.clone()))
    }
    pub fn set_shape(&mut self, shape: DynShape) {
        self.iter_mut().for_each(|tv| tv.set_shape(shape.clone()))
    }
    pub fn rank(&self) -> usize {
        match self.0.as_slice() {
            [only] => only.rank(),
            _ => 0,
        }
    }
    pub fn leading_rank(&self) -> usize {
        match self.0.as_slice() {
            [only] => only.leading_rank(),
            _ => 0,
        }
    }
    pub fn into_row(self) -> Self {
        self.into_iter().map(TypeVal::into_row).collect()
    }
    pub fn into_first_row(self) -> Self {
        self.into_iter().map(TypeVal::into_first_row).collect()
    }
    pub fn into_last_row(self) -> Self {
        self.into_iter().map(TypeVal::into_last_row).collect()
    }
    pub fn prepend_dim(&mut self, dim: Dim) {
        self.iter_mut().for_each(|tv| tv.prepend_dim(dim))
    }
    pub fn boxed(self) -> Self {
        Scalar::Box(ScalarBox::All(Box::new(self.into()))).into()
    }
    pub fn unboxed(self) -> Self {
        match self.0.as_slice() {
            [
                TypeVal::Type(Type {
                    scalar: Scalar::Box(ScalarBox::All(ty)),
                    ..
                }),
            ] => (**ty).clone().into(),
            _ => self.into_iter().map(TypeVal::unboxed).collect(),
        }
    }
    pub fn is_any(&self) -> bool {
        matches!(self.0[0].clone(), TypeVal::Type(ty) if ty.is_any())
    }
    pub fn reshape_scalar(&mut self, dim: Dim, suffix: bool) {
        self.iter_mut()
            .for_each(|tv| tv.reshape_scalar(dim.clone(), suffix.clone()))
    }
}

impl fmt::Display for OrTypeVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_slice() {
            [only] => write!(f, "{only}"),
            tvs => {
                write!(f, "(")?;
                for (i, tv) in tvs.iter().enumerate() {
                    if i != 0 {
                        write!(f, "|")?;
                    }
                    write!(f, "{tv}")?;
                }
                write!(f, ")")
            }
        }
    }
}
