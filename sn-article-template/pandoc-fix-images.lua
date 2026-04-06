function Image(el)
  if el.src:match("%.pdf$") then
    el.src = el.src:gsub("%.pdf$", ".png")
  end
  return el
end
