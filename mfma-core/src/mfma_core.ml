(* mfma_core.ml — MFMA tile computation core (OCaml) *)
(* Compiles to C via ocamlopt -output-obj for HLS pipeline *)

let half_to_float (h : int) : float =
  let sign = (h lsr 15) land 0x1 in
  let exp = (h lsr 10) land 0x1F in
  let mantissa = h land 0x3FF in
  if exp = 0x1F then
    if mantissa = 0 then
      if sign = 0 then Float.infinity else Float.neg_infinity
    else Float.nan
  else if exp = 0 then
    let m = if mantissa = 0 then 0.0 else Float.ldexp (Float.of_int mantissa) (-24) in
    if sign = 0 then m else Float.neg m
  else
    let m = Float.ldexp (Float.of_int (lor mantissa 0x400)) (exp - 15) in
    if sign = 0 then m else Float.neg m

let mfma_tile
    (a_tile : int array)
    (b_tile : int array)
    (c_tile : float array) : float array =
  let acc = Array.copy c_tile in
  for m = 0 to 15 do
    for n = 0 to 15 do
      let mutable acc_val = acc.(m * 16 + n) in
      for k = 0 to 15 do
        let a_val = a_tile.(m * 16 + k) in
        let b_val = b_tile.(k * 16 + n) in
        let va = half_to_float a_val in
        let vb = half_to_float b_val in
        acc_val <-
          if Float.is_nan va || Float.is_nan vb || Float.is_nan acc_val then
            Float.nan
          else
            Float.(va *. vb +. acc_val)
      done;
      acc.(m * 16 + n) <- acc_val
    done
  done;
  acc
